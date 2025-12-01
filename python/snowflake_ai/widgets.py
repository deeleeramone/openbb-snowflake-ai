"""Widget endpoints for Snowflake AI OpenBB Workspace integration."""

# pylint: disable=C0301,E0611,R0912,R0914,R0915,W0613,W0718

import asyncio
import base64
import json
import os
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Body, Depends, HTTPException
from openbb_platform_api.response_models import (
    OmniWidgetResponseModel,
)

from ._snowflake_ai import SnowflakeAI
from .document_processor import DocumentProcessor
from .logger import get_logger


logger = get_logger(__name__)
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

    # Check if this is a remove request (document list provided)
    if documents := payload.get("document", []):
        doc_proc = DocumentProcessor.instance()
        results = []
        for doc_path in documents:
            success, message = await doc_proc.remove_file_from_stage(client, doc_path)
            results.append(
                {
                    "document": doc_path,
                    "removed": success,
                    "message": message,
                }
            )

        return True

    file_bytes: bytes | None = None
    file_name: str = ""
    stage_name = payload.get("stage_name")  # Optional, defaults to CORTEX_UPLOADS
    embed_images = payload.get("embed_images", False)  # Optional, default False

    # Handle different input methods
    if path := payload.get("path", ""):  # nosec
        if not os.path.isfile(path):  # nosec
            raise HTTPException(
                status_code=400, detail=f"File path does not exist: {path}"
            )
        path_obj = os.path.normpath(path)
        new_path = str(path_obj)
        file_name = path.rsplit("/", maxsplit=1)[-1]

        with open(new_path, "rb") as file:
            file_bytes = file.read()

    elif url := payload.get("url", ""):
        file_name = (
            payload.get("file_name") or url.split("/")[-1].split("?")[0]
        )  # Remove query params

        async with await get_async_requests_session() as session:
            async with await session.get(url) as response:
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
        if not file_name.endswith(".pdf"):
            file_name += ".pdf"
        print(file_name)
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
        doc_proc = DocumentProcessor.instance()

        # STEP 1: Extract and store PDF positions/metadata IMMEDIATELY (before returning)
        # This ensures positions/metadata are always available for AI chat
        if file_ext == ".pdf" and file_bytes:
            try:
                # Single-pass extraction of positions + metadata
                _, text_positions, pdf_metadata = doc_proc.extract_pdf_with_positions(
                    file_bytes, extract_metadata=True
                )
                logger.info(
                    "Widget upload: Extracted %d positions and %d outline entries for %s",
                    len(text_positions) if text_positions else 0,
                    len(pdf_metadata.get("outline", [])) if pdf_metadata else 0,
                    file_name,
                )

                # Store positions immediately
                if text_positions:
                    await doc_proc.store_pdf_positions_in_snowflake(
                        client,
                        file_name,
                        stage_path,
                        text_positions,
                    )
                    logger.info(
                        "Widget upload: Stored %d text positions for %s",
                        len(text_positions),
                        file_name,
                    )

                # Store metadata immediately
                if pdf_metadata:
                    await doc_proc.store_document_metadata(
                        client,
                        file_name,
                        stage_path,
                        pdf_metadata,
                    )
                    logger.info(
                        "Widget upload: Stored document metadata for %s", file_name
                    )

            except Exception as e:
                logger.warning(
                    "Widget upload: Failed to extract/store PDF metadata: %s", e
                )

        # STEP 2: Create job record
        job_id = None
        try:
            job_id = await doc_proc.create_processing_job(
                client, file_name, stage_path, embed_images
            )
        except Exception as e:
            logger.warning("Widget upload: Job creation failed: %s", e)

        # STEP 3: Start background task for image upload + stored procedure call
        # This returns immediately - all heavy work happens in background
        asyncio.create_task(
            _process_document_with_images(
                client,
                doc_proc,
                file_name,
                stage_path,
                db_name,
                schema_name,
                embed_images,
                job_id,
                pdf_bytes=file_bytes if embed_images else None,
            )
        )

        processing_status = "parsing"
        job_info = f" (job_id: {job_id})" if job_id else ""
        processing_message = (
            f"File uploaded to {stage_path}. "
            f"Document parsing started in background{job_info}"
            + (" with image embedding" if embed_images else "")
            + ". "
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

    return status


async def _process_document_with_images(
    client: SnowflakeAI,
    doc_proc: DocumentProcessor,
    file_name: str,
    stage_path: str,
    target_database: str,
    target_schema: str,
    embed_images: bool = False,
    job_id: str | None = None,
    pdf_bytes: bytes | None = None,
):
    """Background task to process document with parallel image upload and parsing.

    This runs asynchronously in the background - the upload endpoint returns immediately.

    Flow (PARALLEL):
    1. Start document parsing via stored procedure immediately
    2. Upload images to DOCUMENT_IMAGES stage concurrently (if embed_images=True)
    3. Stored procedure handles text embeddings, then waits for images to generate image embeddings

    Parameters
    ----------
    pdf_bytes : bytes | None
        Raw PDF bytes for image extraction. Only needed if embed_images=True.
    """
    logger.info(
        "_process_document_with_images ENTRY: file=%s, embed_images=%s, has_pdf_bytes=%s",
        file_name,
        embed_images,
        pdf_bytes is not None,
    )

    try:
        # Run image upload and document processing in PARALLEL
        # This is much faster than sequential processing

        async def upload_images_task():
            """Upload images concurrently while parsing runs."""
            if embed_images and pdf_bytes:
                try:
                    count = await doc_proc.upload_images_and_store_metadata(
                        client,
                        file_name,
                        stage_path,
                        pdf_bytes,
                    )
                    logger.info(
                        "_process_document_with_images: Uploaded %d images for %s",
                        count,
                        file_name,
                    )
                    return count
                except Exception as e:
                    logger.warning(
                        "_process_document_with_images: Image upload failed: %s", e
                    )
            return 0

        async def parsing_task():
            """Start document parsing (text parsing + embeddings)."""
            if job_id:
                try:
                    success = await doc_proc.start_document_processing_async(
                        client,
                        job_id,
                        file_name,
                        stage_path,
                        embed_images,
                    )
                    return success
                except Exception as e:
                    logger.error(
                        "_process_document_with_images: Stored procedure failed: %s", e
                    )
                    return False
            return False

        # Run both tasks concurrently
        image_count, parsing_success = await asyncio.gather(
            upload_images_task(),
            parsing_task(),
            return_exceptions=False,
        )

        if not parsing_success and job_id:
            # Fall back to Python background processing
            logger.warning("Stored procedure failed, falling back to Python processing")
            await _process_document_background(
                client,
                stage_path,
                file_name,
                target_database,
                target_schema,
                embed_images,
                job_id,
                pdf_bytes=pdf_bytes,
            )

    except Exception as e:
        logger.error(
            "_process_document_with_images failed for %s: %s",
            file_name,
            e,
            exc_info=True,
        )
        if job_id:
            try:
                await doc_proc.update_processing_job(
                    client, job_id, status="failed", error_message=str(e)[:500]
                )
            except Exception:
                pass


async def _process_document_background(
    client: SnowflakeAI,
    stage_path: str,
    filename: str,
    target_database: str,
    target_schema: str,
    embed_images: bool = False,
    job_id: str | None = None,
    pdf_bytes: bytes | None = None,
):
    """Background task to parse document with Cortex and generate embeddings.

    This runs asynchronously in the background - the upload endpoint returns immediately.
    PDF positions and metadata are extracted and stored during upload (not here).

    Parameters
    ----------
    pdf_bytes : bytes | None
        Raw PDF bytes for image extraction. Only needed if embed_images=True.
    """
    logger.info(
        "_process_document_background ENTRY: file=%s, embed_images=%s, has_pdf_bytes=%s, pdf_len=%d",
        filename,
        embed_images,
        pdf_bytes is not None,
        len(pdf_bytes) if pdf_bytes else 0,
    )
    doc_proc = DocumentProcessor.instance()
    qualified_table = f"{target_database}.{target_schema}.DOCUMENT_PARSE_RESULTS"

    # Update job status to parsing
    if job_id:
        try:
            await doc_proc.update_processing_job(client, job_id, status="parsing")
        except Exception as e:
            logger.warning("Failed to update job status: %s", e)

    # STEP 1: Parse document with Snowflake AI_PARSE_DOCUMENT
    page_count = 0
    try:
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
        await asyncio.to_thread(client.execute_statement, create_table_query)

        # Delete any existing records for this document to avoid duplicates
        escaped_filename = filename.replace("'", "''")
        escaped_stage_path = stage_path.replace("'", "''")
        delete_query = f"""
        DELETE FROM {qualified_table}
        WHERE FILE_NAME = '{escaped_filename}' OR STAGE_PATH = '{escaped_stage_path}'
        """
        await asyncio.to_thread(client.execute_statement, delete_query)
        logger.info("Widget upload: Deleted existing parse results for %s", filename)

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
        await asyncio.to_thread(client.execute_statement, insert_query)

        logger.info("Widget upload: Document parsing completed for %s", filename)

        # Get page count
        count_query = f"""
        SELECT COUNT(*) as cnt FROM {qualified_table}
        WHERE FILE_NAME = '{filename.replace("'", "''")}' 
          AND STAGE_PATH = '{stage_path.replace("'", "''")}'
        """
        count_result = await asyncio.to_thread(client.execute_query, count_query)
        count_data = json.loads(count_result) if count_result else {}
        if count_data.get("rowData"):
            row = count_data["rowData"][0]
            page_count = row.get("CNT", row.get("cnt", 0))

        # Update job with parse completion
        if job_id:
            try:
                await doc_proc.update_processing_job(
                    client,
                    job_id,
                    status="embedding",
                    last_completed_step="parse",
                    page_count=page_count,
                )
            except Exception as e:
                logger.warning("Failed to update job status: %s", e)

    except Exception as e:
        logger.error("Widget upload: Background parsing failed for %s: %s", filename, e)
        if job_id:
            try:
                await doc_proc.update_processing_job(
                    client,
                    job_id,
                    status="failed",
                    error_message=str(e)[:500],
                )
            except Exception:
                pass
        return

    # STEP 2: Generate embeddings after parsing
    try:

        # Ensure embeddings table exists
        await doc_proc._create_document_embeddings_table(client)

        # Get parsed pages
        select_query = f"""
        SELECT PAGE_NUMBER, PAGE_CONTENT
        FROM {qualified_table}
        WHERE FILE_NAME = '{filename.replace("'", "''")}' 
          AND STAGE_PATH = '{stage_path.replace("'", "''")}'
        ORDER BY PAGE_NUMBER
        """
        result_json = await asyncio.to_thread(client.execute_statement, select_query)
        rows = json.loads(result_json) if result_json else []

        pages = []
        for row in rows:
            page_num = row.get("PAGE_NUMBER") or row.get("page_number")
            content = row.get("PAGE_CONTENT") or row.get("page_content")
            if page_num and content:
                pages.append({"page": int(page_num), "content": content})

        if pages:
            logger.info(
                "Widget upload: Generating embeddings for %s (embed_images=%s, has_pdf_bytes=%s)",
                filename,
                embed_images,
                pdf_bytes is not None,
            )

            # Generate embeddings - pdf_bytes passed from upload for image extraction
            success = await doc_proc._generate_embeddings_for_document(
                client=client,
                file_name=filename,
                stage_path=stage_path,
                pages=pages,
                pdf_bytes=pdf_bytes,
                embed_images=embed_images,
            )

            if success:
                logger.info(
                    "Widget upload: Embeddings generated successfully for %s",
                    filename,
                )
                # Update job as completed
                if job_id:
                    try:
                        await doc_proc.update_processing_job(
                            client,
                            job_id,
                            status="completed",
                            last_completed_step="embed",
                            embedding_count=len(pages),
                        )
                    except Exception as e:
                        logger.warning("Failed to update job status: %s", e)
            else:
                logger.warning(
                    "Widget upload: Failed to generate embeddings for %s", filename
                )
                if job_id:
                    try:
                        await doc_proc.update_processing_job(
                            client,
                            job_id,
                            status="completed",
                            last_completed_step="embed",
                            error_message="Embedding generation returned False",
                        )
                    except Exception:
                        pass
        else:
            # No pages to embed, mark as completed
            if job_id:
                try:
                    await doc_proc.update_processing_job(
                        client,
                        job_id,
                        status="completed",
                        last_completed_step="parse",
                    )
                except Exception:
                    pass

    except Exception as e:
        logger.error(
            "Widget upload: Embedding generation failed for %s: %s", filename, e
        )
        if job_id:
            try:
                await doc_proc.update_processing_job(
                    client,
                    job_id,
                    status="failed",
                    error_message=f"Embedding failed: {str(e)[:400]}",
                )
            except Exception:
                pass


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


@router.get("/processing_status/{job_id}")
async def get_processing_status(
    job_id: str,
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
) -> dict:
    """Get the status of a document processing job.

    Parameters
    ----------
    job_id : str
        The job ID returned from upload_widget_file

    Returns
    -------
    dict
        Job status including:
        - job_id: The job ID
        - file_name: Name of the file being processed
        - stage_path: Stage path where the file is stored
        - status: Current status (pending, parsing, embedding, completed, failed)
        - last_completed_step: Name of the last completed processing step
        - page_count: Number of pages parsed
        - embedding_count: Number of embeddings generated
        - error_message: Error message if job failed
        - created_at: When the job was created
        - updated_at: When the job was last updated
    """
    doc_proc = DocumentProcessor.instance()
    status = await doc_proc.get_processing_job_status(client, job_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return status


@router.get("/document_ready")
async def check_document_ready(
    stage_path: str,
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
) -> dict:
    """Check what data is available for a document.

    Use this to determine if a document is ready for AI chat and what
    context can be provided.

    Parameters
    ----------
    stage_path : str
        The stage path of the document to check

    Returns
    -------
    dict
        Availability status including:
        - has_positions: Whether text positions are available
        - has_metadata: Whether document metadata is available
        - has_parsed_pages: Whether Cortex-parsed pages are available
        - has_embeddings: Whether embeddings are available
        - page_count: Number of parsed pages
        - embedding_count: Number of embeddings
        - processing_status: Current processing status from job table (if exists)
    """
    doc_proc = DocumentProcessor.instance()
    return await doc_proc.check_document_ready(client, stage_path)


@lru_cache(maxsize=128)
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
            logger.error("Failed to download document %s: %s", filename, exc)
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
