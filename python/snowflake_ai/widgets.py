"""Widget endpoints for Snowflake AI OpenBB Workspace integration."""

# pylint: disable=C0301,E0611,R0912,R0914,R0915,W0613,W0718

import asyncio
import base64
import json
import os
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from openbb_platform_api.response_models import (
    OmniWidgetResponseModel,
)

from ._snowflake_ai import SnowflakeAI

router = APIRouter(prefix="/widgets", tags=["widgets"])


class SnowflakeClient:
    """Lazily instantiate and reuse a SnowflakeAI connection for widget calls."""

    def __init__(self):
        self._client: SnowflakeAI | None = None

    def _build_client(self) -> SnowflakeAI:
        required_env = {
            "SNOWFLAKE_USER": os.environ.get("SNOWFLAKE_USER"),
            "SNOWFLAKE_PASSWORD": os.environ.get("SNOWFLAKE_PASSWORD"),
            "SNOWFLAKE_ACCOUNT": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_ROLE": os.environ.get("SNOWFLAKE_ROLE"),
        }
        missing = [key for key, value in required_env.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required Snowflake credentials: " + ", ".join(sorted(missing))
            )

        return SnowflakeAI(
            user=required_env["SNOWFLAKE_USER"],
            password=required_env["SNOWFLAKE_PASSWORD"],
            account=required_env["SNOWFLAKE_ACCOUNT"],
            role=required_env["SNOWFLAKE_ROLE"],
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or "",
            database=os.environ.get("SNOWFLAKE_DATABASE") or "",
            schema=os.environ.get("SNOWFLAKE_SCHEMA") or "",
        )

    def __call__(self) -> SnowflakeAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def close(self) -> None:
        """Close the Snowflake client connection."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


snowflake_client = SnowflakeClient()


def close_widget_client() -> None:
    """Close the Snowflake client on shutdown."""
    snowflake_client.close()


router.add_event_handler("shutdown", close_widget_client)


@router.post("/execute", response_model=OmniWidgetResponseModel)
def execute_widget_query(
    payload: dict,
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
):
    """Execute a Snowflake SQL query."""
    try:
        raw_result = client.execute_query(payload.get("prompt", ""))
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        parsed_result = json.loads(raw_result)
    except json.JSONDecodeError:
        parsed_result = raw_result

    row_data = parsed_result.get("rowData", [])

    if not row_data:
        raise HTTPException(
            status_code=500, detail="No row data returned from Snowflake query."
        )

    return {"content": row_data}


@router.post("/upload")
async def upload_widget_file(
    payload: Annotated[dict, Body()],
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
):
    """Upload a document file to Snowflake for Cortex analysis.

    Expected payload (one of):
    - {"path": "/local/path/to/file.pdf"}
    - {"url": "https://example.com/file.pdf"}
    - {"file_name": "doc.pdf", "content": "<base64-encoded-bytes>"}
    - {"document": ["@STAGE/file.pdf"], "submit": true}  # Remove document

    Supported file types: PDF, DOCX, TXT, HTML, JSON, CSV, XML

    Post-upload processing:
    - PDFs/Docs: Parsed via Cortex AI and stored in DOCUMENT_PARSE_RESULTS
    - JSON/XML: Loaded into a Snowflake table as VARIANT data
    - CSV: Loaded into a Snowflake table with inferred schema

    Returns the stage path where the file was uploaded.
    """
    # pylint: disable=import-outside-toplevel
    from openbb_core.provider.utils.helpers import get_async_requests_session
    from .helpers import remove_file_from_stage

    # Check if this is a remove request (document list provided)
    if documents := payload.get("document", []):
        results = []
        for doc_path in documents:
            success, message = await remove_file_from_stage(client, doc_path)
            results.append(
                {
                    "document": doc_path,
                    "removed": success,
                    "message": message,
                }
            )

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Upload widget remove results: {results}")

        return True

    file_bytes: bytes | None = None
    file_name: str = ""
    stage_name = payload.get("stage_name")  # Optional, defaults to CORTEX_UPLOADS

    # Handle different input methods
    if path := payload.get("path", ""):
        if not os.path.isfile(path):
            raise HTTPException(
                status_code=400, detail=f"File path does not exist: {path}"
            )
        path_obj = os.path.normpath(path)
        new_path = str(path_obj)
        file_name = path.rsplit("/", maxsplit=1)[-1]

        with open(new_path, "rb") as file:
            file_bytes = file.read()

    elif url := payload.get("url", ""):
        file_name = url.split("/")[-1].split("?")[0]  # Remove query params

        async with await get_async_requests_session() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                file_bytes = await response.read()

    elif content_b64 := payload.get("content", ""):
        file_name = payload.get("file_name", "")
        if not file_name:
            raise HTTPException(
                status_code=400, detail="Missing file_name when using content."
            )
        try:
            file_bytes = base64.b64decode(content_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid base64 content: {exc}"
            ) from exc
    else:
        raise HTTPException(
            status_code=400, detail="Missing 'path', 'url', or 'content' in request."
        )

    if not file_name:
        raise HTTPException(status_code=400, detail="Could not determine file name.")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="No file content to upload.")

    try:
        # Upload bytes to Snowflake stage
        stage_path = client.upload_bytes_to_stage(
            list(file_bytes),  # Convert bytes to list for PyO3
            file_name,
            stage_name,
        )
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Determine file type and trigger appropriate post-processing
    file_ext = os.path.splitext(file_name)[1].lower()

    # Get user schema for storage
    snowflake_user = client.get_current_user()
    sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
    schema_name = f"USER_{sanitized_user}".upper()
    db_name = "OPENBB_AGENTS"

    processing_status = "uploaded"
    processing_message = f"File uploaded successfully to {stage_path}"

    # Parseable document types (use Cortex AI parsing)
    parseable_extensions = {
        ".pdf",
        ".txt",
        ".docx",
        ".doc",
        ".rtf",
        ".md",
        ".html",
        ".htm",
    }
    # Semi-structured data types (load into tables)
    data_extensions = {".json", ".xml"}
    # Tabular data types
    tabular_extensions = {".csv", ".tsv"}

    if file_ext in parseable_extensions:
        # Fire and forget background document parsing
        asyncio.create_task(
            _process_document_background(
                client, stage_path, file_name, db_name, schema_name
            )
        )
        processing_status = "parsing"
        processing_message = (
            f"File uploaded to {stage_path}. "
            f"Document parsing started in background. "
            f"Results will be saved to {db_name}.{schema_name}.DOCUMENT_PARSE_RESULTS"
        )

    elif file_ext in data_extensions:
        # Load JSON/XML into a table
        try:
            table_name = "".join(
                c if c.isalnum() else "_" for c in os.path.splitext(file_name)[0]
            ).upper()
            qualified_table = f"{db_name}.{schema_name}.{table_name}"
            file_type = "JSON" if file_ext == ".json" else "XML"

            # Create file format
            format_name = f"{db_name}.{schema_name}.FORMAT_{table_name}"
            if file_type == "XML":
                format_query = f"CREATE OR REPLACE FILE FORMAT {format_name} TYPE = {file_type} STRIP_OUTER_ELEMENT = TRUE"
            else:
                format_query = f"CREATE OR REPLACE FILE FORMAT {format_name} TYPE = {file_type} STRIP_OUTER_ARRAY = TRUE"
            client.execute_statement(format_query)

            # Create table
            client.execute_statement(
                f"CREATE OR REPLACE TABLE {qualified_table} (RAW_DATA VARIANT)"
            )

            # Load data
            client.execute_statement(
                f"COPY INTO {qualified_table} FROM '{stage_path}' "
                f"FILE_FORMAT = (FORMAT_NAME = {format_name}) ON_ERROR = 'CONTINUE'"
            )

            # Get row count
            count_result = client.execute_query(
                f"SELECT COUNT(*) as cnt FROM {qualified_table}"
            )
            count_data = json.loads(count_result)
            num_rows = 0
            if count_data.get("rowData"):
                row = count_data["rowData"][0]
                num_rows = row.get("CNT", row.get("cnt", 0))

            processing_status = "loaded"
            processing_message = (
                f"File uploaded and loaded into table {qualified_table} "
                f"with {num_rows} rows."
            )
        except Exception as e:
            processing_status = "upload_only"
            processing_message = (
                f"File uploaded to {stage_path}, but failed to load into table: {e}"
            )

    elif file_ext in tabular_extensions:
        # Load CSV/TSV into a table with inferred schema
        try:
            table_name = "".join(
                c if c.isalnum() else "_" for c in os.path.splitext(file_name)[0]
            ).upper()
            qualified_table = f"{db_name}.{schema_name}.{table_name}"

            # Create file format for CSV
            format_name = f"{db_name}.{schema_name}.FORMAT_{table_name}"
            delimiter = "\\t" if file_ext == ".tsv" else ","
            client.execute_statement(
                f"CREATE OR REPLACE FILE FORMAT {format_name} "
                f"TYPE = CSV FIELD_DELIMITER = '{delimiter}' "
                f"SKIP_HEADER = 1 FIELD_OPTIONALLY_ENCLOSED_BY = '\"'"
            )

            # Use INFER_SCHEMA to create table with proper columns
            client.execute_statement(
                f"CREATE OR REPLACE TABLE {qualified_table} "
                f"USING TEMPLATE (SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*)) "
                f"FROM TABLE(INFER_SCHEMA(LOCATION => '{stage_path}', "
                f"FILE_FORMAT => {format_name})))"
            )

            # Load data
            client.execute_statement(
                f"COPY INTO {qualified_table} FROM '{stage_path}' "
                f"FILE_FORMAT = (FORMAT_NAME = {format_name}) "
                f"MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE ON_ERROR = 'CONTINUE'"
            )

            # Get row count
            count_result = client.execute_query(
                f"SELECT COUNT(*) as cnt FROM {qualified_table}"
            )
            count_data = json.loads(count_result)
            num_rows = 0
            if count_data.get("rowData"):
                row = count_data["rowData"][0]
                num_rows = row.get("CNT", row.get("cnt", 0))

            processing_status = "loaded"
            processing_message = (
                f"File uploaded and loaded into table {qualified_table} "
                f"with {num_rows} rows."
            )
        except Exception as e:
            processing_status = "upload_only"
            processing_message = (
                f"File uploaded to {stage_path}, but failed to load into table: {e}"
            )

    status = {
        "success": True,
        "file_name": file_name,
        "stage_path": stage_path,
        "processing_status": processing_status,
        "message": processing_message,
    }
    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Upload widget status: {status}")

    return True


async def _process_document_background(
    client: SnowflakeAI,
    stage_path: str,
    filename: str,
    target_database: str,
    target_schema: str,
):
    """Background task to parse document and save results."""

    def _run_sync():
        try:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Widget upload: Starting background parsing for {filename}"
                )

            # Ensure results table exists
            qualified_table = (
                f"{target_database}.{target_schema}.DOCUMENT_PARSE_RESULTS"
            )
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                FILE_NAME STRING,
                STAGE_PATH STRING,
                PAGE_NUMBER INTEGER,
                PAGE_CONTENT STRING,
                METADATA VARIANT,
                PARSED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            client.execute_statement(create_table_query)

            # Extract stage info from stage_path
            clean_path = stage_path.lstrip("@")
            if "/" in clean_path:
                parts = clean_path.split("/", 1)
                stage_name_extracted = f"@{parts[0]}"
                relative_file_path = parts[1]
            else:
                stage_name_extracted = f"@{clean_path}"
                relative_file_path = filename

            # Parse and insert in a single query
            insert_query = f"""
            INSERT INTO {qualified_table} (FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, METADATA)
            WITH DOC AS (
              SELECT AI_PARSE_DOCUMENT(TO_FILE('{stage_name_extracted}', '{relative_file_path}'), {{'mode': 'LAYOUT', 'page_split': true}}) AS RAW
            )
            SELECT 
                '{filename}',
                '{stage_path}',
                INDEX + 1,
                GET(VALUE, 'content')::STRING,
                GET(DOC.RAW, 'metadata')
            FROM DOC, TABLE(FLATTEN(input => COALESCE(GET(DOC.RAW, 'pages'), DOC.RAW)))
            """
            client.execute_statement(insert_query)

            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Widget upload: Background parsing complete for {filename}"
                )

        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[ERROR] Widget upload: Background parsing failed for {filename}: {e}"
                )

    # Run synchronous code in thread pool
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run_sync)


@router.get("/list_documents")
def list_cortex_documents(
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
    as_choices: bool = False,
) -> list:
    """List uploaded documents in Snowflake Cortex.

    Returns a list of documents with their metadata:
    - file_name: The name of the file
    - stage_path: Where the file is stored in Snowflake
    - is_parsed: Whether the document has been parsed by Cortex
    - page_count: Number of parsed pages (0 if not parsed)
    """
    try:
        # Returns list of (file_name, stage_path, is_parsed, page_count) tuples
        documents_raw = client.list_cortex_documents()

        # Convert to list of dicts for better JSON response
        documents = [
            {
                "file_name": doc[0],
                "stage_path": doc[1],
                "is_parsed": doc[2],
                "page_count": doc[3],
            }
            for doc in documents_raw
        ]
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if as_choices:
        # Return as list of choice dicts for UI dropdowns
        choices = []
        choices.extend(
            [
                {"label": doc["file_name"], "value": doc["stage_path"]}
                for doc in documents
            ]
        )
        return choices

    return documents


@router.get("/list_document_choices", include_in_schema=False)
def list_cortex_document_choices(
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
    show_summary: bool = False,
):
    """List uploaded documents in Snowflake Cortex as choice dicts for UI dropdowns."""
    choices = list_cortex_documents(client=client, as_choices=True)
    if show_summary:
        choices.insert(0, {"label": "Summary", "value": "summary"})

    return choices


@router.post("/download_document")
def download_cortex_document(
    payload: Annotated[dict, Body()],
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
):
    """Download documents from Snowflake Cortex.

    Supports: PDF, DOCX, TXT, HTML, JSON, CSV, XML

    Returns file content with appropriate data_format based on file extension.
    """
    # Map file extensions to data types
    extension_to_data_type = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".txt": "text",
        ".text": "text",
        ".html": "html",
        ".htm": "html",
        ".json": "json",
        ".csv": "csv",
        ".xml": "xml",
        ".xlsx": "spreadsheet",
        ".xls": "spreadsheet",
    }

    documents = payload.get("document", [])
    files = []

    for name in documents:
        if name == "summary":
            document_list = list_cortex_documents(client=client)

            # Build table with left-justified filename and right-aligned other columns
            stream_content = "BT\n/F1 10 Tf\n"
            y = 750
            right_margin = 560
            char_width = 6

            # Title
            stream_content += (
                f"1 0 0 1 50 {y} Tm\n(AVAILABLE DOCUMENTS IN YOUR STAGE) Tj\n"
            )
            y -= 28

            # Header row - FILE NAME
            stream_content += f"1 0 0 1 50 {y} Tm\n(FILE NAME) Tj\n"
            # "CORTEX-PARSED"
            stream_content += (
                f"1 0 0 1 {480 - 13 * char_width} {y} Tm\n(CORTEX-PARSED) Tj\n"
            )
            # "PAGES"
            stream_content += (
                f"1 0 0 1 {right_margin - 5 * char_width} {y} Tm\n(PAGES) Tj\n"
            )
            y -= 14

            # Separator line - 85 dashes from margin to margin
            dash_count = (right_margin - 50) // char_width
            stream_content += f"1 0 0 1 50 {y} Tm\n({'-' * dash_count}) Tj\n"
            y -= 14

            # Data rows
            for doc in document_list:
                fname = doc["file_name"][:50]
                escaped_fname = (
                    fname.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
                )
                parsed = "yes" if doc["is_parsed"] else "no"
                pages = str(doc["page_count"])
                # Filename left-aligned
                stream_content += f"1 0 0 1 50 {y} Tm\n({escaped_fname}) Tj\n"
                # Parsed status right-aligned
                stream_content += (
                    f"1 0 0 1 {480 - len(parsed) * char_width} {y} Tm\n({parsed}) Tj\n"
                )
                # Page count right-aligned
                stream_content += f"1 0 0 1 {right_margin - len(pages) * char_width} {y} Tm\n({pages}) Tj\n"
                y -= 14

            # Footer
            y -= 14
            stream_content += f"1 0 0 1 50 {y} Tm\n(To add or remove documents, click 'Open Form'.) Tj\n"
            stream_content += "ET"

            stream_len = len(stream_content.encode("latin-1"))

            pdf_content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {stream_len} >> stream
{stream_content}
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
0
%%EOF"""

            pdf_base64 = base64.b64encode(pdf_content.encode("latin-1")).decode("utf-8")

            files.append(
                {
                    "content": pdf_base64,
                    "data_format": {
                        "data_type": "pdf",
                        "filename": "summary.pdf",
                    },
                }
            )
            return files

        document_id = name
        filename = document_id.split("/")[-1]

        # Determine data type from extension
        ext = os.path.splitext(filename)[1].lower()
        data_type = extension_to_data_type.get(ext, "raw")

        try:
            document_content = client.download_cortex_document(document_id)
        except Exception as exc:
            files.append(
                {
                    "error_type": "download_error",
                    "content": f"Failed to download {filename}",
                }
            )
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[ERROR] Failed to download document {filename}: {exc}")
            continue

        if not document_content:
            files.append(
                {
                    "error_type": "not_found",
                    "content": f"Document '{filename}' not found or empty",
                }
            )
            continue

        files.append(
            {
                "content": document_content,
                "data_format": {
                    "data_type": data_type,
                    "filename": filename,
                },
            }
        )

    return files
