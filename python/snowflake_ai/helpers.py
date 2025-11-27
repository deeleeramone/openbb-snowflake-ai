"""Snowflake AI helpers."""

# flake8: noqa: PLR0911, PLR0912
# pylint: disable = R0911, R0912, R0914, R0915, R0917, C0103, C0415

import base64
import hashlib
import io
import json
import os
import re
import threading
from collections import defaultdict
from typing import Any, Iterable, Mapping, Tuple

from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    ImageDataFormat,
    PdfDataFormat,
    RawObjectDataFormat,
    SingleDataContent,
    SingleFileReference,
    SpreadsheetDataFormat,
)

# Store PDF text positions as dictionaries (from pdfplumber extraction)
pdf_text_blocks: dict[str, list[dict]] = {}
# Store which quotes the LLM actually referenced
llm_referenced_quotes: dict[str, list[tuple[str, int]]] = {}

# Track document sources for unified citations (widget PDFs and Snowflake documents)
# Structure: {conversation_id: {"source_type": "widget"|"snowflake", "file_name": str, "stage_path": str|None}}
document_sources: dict[str, dict] = {}

# Store Snowflake document page content for citation matching
# Structure: {conversation_id: [{page: int, content: str, file_name: str}]}
snowflake_document_pages: dict[str, list[dict]] = {}

# Conversation deduplication caches (in-memory only)
_MESSAGE_SIGNATURE_CACHE: dict[str, set[str]] = defaultdict(set)
_SIGNATURE_LOCK = threading.Lock()


def extract_filename_from_stage_path(stage_path: str) -> str | None:
    """
    Extract the filename from a Snowflake stage path.

    Examples:
        "@OPENBB_AGENTS.USER_DLEE.CORTEX_UPLOADS/technology-investment.pdf"
        -> "technology-investment.pdf"

        "@MYDB.MYSCHEMA.MYSTAGE/subfolder/document.pdf"
        -> "document.pdf"

    Args:
        stage_path: The stage path (may or may not start with @)

    Returns:
        The filename extracted from the path, or None if not found
    """
    if not stage_path:
        return None

    # Remove leading @ if present
    path = stage_path.lstrip("@")

    # Split by / and get the last part
    if "/" in path:
        return path.split("/")[-1]

    return None


def extract_filename_from_widget(widget) -> str | None:
    """
    Extract the actual document filename from a widget's params.

    Looks for params with current_value containing a stage path
    (like @SCHEMA.STAGE/filename.pdf) and extracts the filename.

    Args:
        widget: A widget object with params attribute

    Returns:
        The extracted filename, or None if not found
    """
    if not hasattr(widget, "params") or not widget.params:
        return None

    for param in widget.params:
        current_value = getattr(param, "current_value", None)
        if current_value and isinstance(current_value, str):
            # Check if it looks like a stage path
            if current_value.startswith("@") and "/" in current_value:
                filename = extract_filename_from_stage_path(current_value)
                if filename and (filename.endswith(".pdf") or "." in filename):
                    return filename

    return None


def _normalize_content(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _prepare_metadata(
    role: str | None,
    content: str | None,
    metadata: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict | None:
    """Merge metadata, details, and inferred attributes for signature building."""

    merged: dict[str, Any] = {}
    if isinstance(details, Mapping):
        for key, value in details.items():
            if value is not None:
                merged[key] = value

    if isinstance(metadata, Mapping):
        for key, value in metadata.items():
            if value is not None:
                merged[key] = value

    normalized_content = (content or "").strip()
    role = (role or "").lower()

    if normalized_content.startswith("[Tool Result from "):
        tool_fragment = normalized_content[len("[Tool Result from ") :]
        tool_name = tool_fragment.split("]", 1)[0]
        merged.setdefault("message_type", "tool_result")
        merged.setdefault("tool_name", tool_name)
    elif normalized_content.startswith("[Tool Result"):
        merged.setdefault("message_type", "tool_result")

    if merged.get("is_tool_result") and merged.get("message_type") is None:
        merged["message_type"] = "tool_result"

    if role == "assistant":
        merged.setdefault("message_type", "assistant")
    elif role in {"human", "user"}:
        merged.setdefault("message_type", "user")

    return merged or None


def _serialize_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not metadata:
        return None
    serialized: dict[str, Any] = {}
    for key in sorted(metadata):
        value = metadata[key]
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool, list, dict)):
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def build_message_signature(
    role: str | None,
    content: str | None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Create a stable hash for a message using role, normalized content, and metadata."""

    payload = {
        "role": (role or "").strip().lower(),
        "content": _normalize_content(content),
        "metadata": _serialize_metadata(metadata),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def seed_message_signatures(
    conv_id: str, messages: Iterable[Mapping[str, Any]]
) -> None:
    """Prime the signature cache with existing conversation history."""

    with _SIGNATURE_LOCK:
        signature_set = _MESSAGE_SIGNATURE_CACHE.setdefault(conv_id, set())
        for message in messages:
            role = message.get("role") if isinstance(message, Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            details = message.get("details") if isinstance(message, Mapping) else None
            metadata = _prepare_metadata(role, content, details=details)
            signature_set.add(build_message_signature(role, content, metadata))


def should_store_message(
    conv_id: str,
    role: str,
    content: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
) -> bool:
    """Check whether a message is new before persisting it."""

    merged_metadata = _prepare_metadata(
        role, content, metadata=metadata, details=details
    )
    signature = build_message_signature(role, content, merged_metadata)
    with _SIGNATURE_LOCK:
        signature_set = _MESSAGE_SIGNATURE_CACHE.setdefault(conv_id, set())
        if signature in signature_set:
            return False
        signature_set.add(signature)
        return True


def clear_message_signatures(conv_id: str) -> None:
    """Remove all cached signatures for a conversation (e.g., after /clear)."""

    with _SIGNATURE_LOCK:
        _MESSAGE_SIGNATURE_CACHE.pop(conv_id, None)


def get_row_value(
    row: Mapping[str, Any] | None,
    *keys: str,
    default: Any | None = None,
) -> Any | None:
    """Fetch a value from a Snowflake row mapping, matching keys case-insensitively."""

    if not isinstance(row, Mapping):
        return default

    lowered = {str(k).lower(): v for k, v in row.items()}

    for key in keys:
        if not key:
            continue
        lookup_key = key.lower()
        if lookup_key in lowered:
            value = lowered[lookup_key]
            if value not in (None, ""):
                return value
            if value is not None:
                return value

    return default


def split_reasoning_and_final(text: str) -> tuple[str | None, str]:
    separators = ["\n---\n", "\r\n---\r\n", "\n---", "---\n"]
    for sep in separators:
        if sep in text:
            reasoning, final = text.split(sep, 1)
            return reasoning.strip(), final.strip()
    return None, text.strip()


def format_thinking_block(content: str) -> str:
    """Wrap reasoning text in a thinking code block."""
    if not content or not content.strip():
        return ""
    return f"```thinking\n{content.strip()}\n```"


async def upload_pdf_bytes_to_snowflake(
    client,
    pdf_bytes: bytes,
    filename: str,
    conversation_id: str,
) -> tuple[bool, str, str | None]:
    """
    Upload PDF bytes to Snowflake stage and trigger parsing.

    Returns:
        Tuple of (success, message, stage_path)
    """
    import asyncio
    import shutil
    import tempfile
    from pathlib import Path

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        # Get user-specific database and schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()
        stage_name = "CORTEX_UPLOADS"

        # Ensure database exists
        try:
            await run_in_thread(
                client.execute_statement, f"CREATE DATABASE IF NOT EXISTS {db_name}"
            )
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Could not create database: {e}")

        # Ensure schema exists
        try:
            await run_in_thread(
                client.execute_statement,
                f"CREATE SCHEMA IF NOT EXISTS {db_name}.{schema_name}",
            )
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Could not create schema: {e}")

        qualified_stage_name = f'"{db_name}"."{schema_name}"."{stage_name}"'

        # Ensure stage exists
        try:
            create_stage_query = f"""
            CREATE STAGE IF NOT EXISTS {qualified_stage_name}
            DIRECTORY = (ENABLE = TRUE)
            ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
            """
            await run_in_thread(client.execute_statement, create_stage_query)
            await run_in_thread(
                client.execute_statement, f"ALTER STAGE {qualified_stage_name} REFRESH"
            )
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Could not ensure stage: {e}")

        # Write bytes to temp file and upload
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp(prefix="snowflake_pdf_upload_")
            temp_file_path = Path(temp_dir) / filename

            with open(temp_file_path, "wb") as f:
                f.write(pdf_bytes)

            # Upload using the Rust client's method
            result = await run_in_thread(
                client.upload_file_to_stage,
                str(temp_file_path),
                stage_name,
            )

            stage_path = f"@{db_name}.{schema_name}.{stage_name}/{filename}"

            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] PDF uploaded to Snowflake: {stage_path}")

            # Store document source info for citations
            document_sources[conversation_id] = {
                "source_type": "snowflake",
                "file_name": filename,
                "stage_path": stage_path,
                "database": db_name,
                "schema": schema_name,
            }

            # Trigger background parsing
            asyncio.create_task(
                _parse_document_and_store_pages(
                    client, stage_path, filename, conversation_id, db_name, schema_name
                )
            )

            return True, f"PDF uploaded to {stage_path}", stage_path

        finally:
            if temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    except Exception as e:
        error_msg = f"Failed to upload PDF to Snowflake: {e}"
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] {error_msg}")
            import traceback

            traceback.print_exc()
        return False, error_msg, None


async def remove_file_from_stage(
    client,
    file_path: str,
) -> tuple[bool, str]:
    """
    Remove a file from Snowflake stage.

    Args:
        client: The Snowflake client
        file_path: The file path to remove. Can be:
            - Full stage path: "@OPENBB_AGENTS.USER_DLEE.CORTEX_UPLOADS/filename.pdf.gz"
            - Short form: "filename.pdf.gz" (will use user's default stage)
            - Relative path: "CORTEX_UPLOADS/filename.pdf.gz"

    Returns:
        Tuple of (success, message)
    """
    import asyncio

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        # Get user-specific schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()
        stage_name = "CORTEX_UPLOADS"

        # Normalize the file path
        if file_path.startswith("@"):
            # Full stage path provided
            stage_path = file_path
        elif "/" in file_path:
            # Relative path with stage name
            stage_path = f"@{db_name}.{schema_name}.{file_path}"
        else:
            # Just filename
            stage_path = f"@{db_name}.{schema_name}.{stage_name}/{file_path}"

        # Execute REMOVE command - quote the path to handle filenames with spaces
        remove_query = f"REMOVE '{stage_path}'"

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Removing file from stage: {remove_query}")

        result = await run_in_thread(client.execute_statement, remove_query)

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Remove result: {result}")

        # Also delete from DOCUMENT_PARSE_RESULTS if exists
        # Extract filename from the path
        filename = stage_path.split("/")[-1]
        # Remove .gz extension if present for matching
        base_filename = filename.rstrip(".gz")

        qualified_table = f'"{db_name}"."{schema_name}"."DOCUMENT_PARSE_RESULTS"'
        escaped_filename = base_filename.replace("'", "''")
        escaped_filename_gz = filename.replace("'", "''")

        delete_query = f"""
        DELETE FROM {qualified_table}
        WHERE FILE_NAME = '{escaped_filename}' OR FILE_NAME = '{escaped_filename_gz}'
        """

        try:
            await run_in_thread(client.execute_statement, delete_query)
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Deleted parsed content for {filename}")
        except Exception as delete_err:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] No parsed content to delete or error: {delete_err}")

        return True, f"File removed: {stage_path}"

    except Exception as e:
        error_msg = f"Failed to remove file from stage: {e}"
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] {error_msg}")
            import traceback

            traceback.print_exc()
        return False, error_msg


def remove_file_from_stage_sync(
    client,
    file_path: str,
) -> tuple[bool, str]:
    """Synchronous wrapper for remove_file_from_stage."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(
            remove_file_from_stage(client, file_path),
            loop,
        )
        return future.result(timeout=30)
    except RuntimeError:
        return asyncio.run(remove_file_from_stage(client, file_path))
    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Error in sync wrapper: {e}")
        return False, str(e)


async def _parse_document_and_store_pages(
    client,
    stage_path: str,
    filename: str,
    conversation_id: str,
    target_database: str,
    target_schema: str,
):
    """Background task to parse document and store pages for citation matching."""
    import asyncio

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Starting background parsing for {filename}")

        # Create results table
        qualified_table = (
            f'"{target_database}"."{target_schema}"."DOCUMENT_PARSE_RESULTS"'
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
        await run_in_thread(client.execute_statement, create_table_query)

        # Extract stage name and relative file path
        clean_path = stage_path.lstrip("@")
        if "/" in clean_path:
            parts = clean_path.split("/", 1)
            stage_name_extracted = f"@{parts[0]}"
            relative_file_path = parts[1]
        else:
            stage_name_extracted = f"@{clean_path}"
            relative_file_path = filename

        # Parse and insert using Cortex AI_PARSE_DOCUMENT
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
        await run_in_thread(client.execute_statement, insert_query)

        # Fetch pages for citation matching
        select_query = f"""
        SELECT PAGE_NUMBER, PAGE_CONTENT 
        FROM {qualified_table} 
        WHERE FILE_NAME = '{filename}' AND STAGE_PATH = '{stage_path}'
        ORDER BY PAGE_NUMBER
        """
        result_json = await run_in_thread(client.execute_statement, select_query)
        rows = json.loads(result_json) if result_json else []

        # Store pages for citation matching
        pages = []
        for row in rows:
            page_num = get_row_value(row, "PAGE_NUMBER", "page_number")
            content = get_row_value(row, "PAGE_CONTENT", "page_content")
            if page_num and content:
                pages.append(
                    {
                        "page": int(page_num),
                        "content": content,
                        "file_name": filename,
                    }
                )

        snowflake_document_pages[conversation_id] = pages

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(
                f"[DEBUG] Stored {len(pages)} pages for conversation {conversation_id}"
            )

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[ERROR] Background parsing failed for {filename}: {e}")
            import traceback

            traceback.print_exc()


def trigger_snowflake_upload_for_widget_pdf(client, conversation_id: str):
    """
    Check if there's a widget PDF that should be uploaded to Snowflake.
    Triggers background upload if PDF bytes are stored in document_sources.

    This should be called after parse_widget_data to optionally upload
    widget PDFs to Snowflake for persistent storage and Cortex parsing.
    """
    import asyncio

    source_info = document_sources.get(conversation_id)
    if not source_info:
        return

    if source_info.get("source_type") != "widget":
        return

    pdf_bytes = source_info.get("pdf_bytes")
    if not pdf_bytes:
        return

    # Check if already uploaded
    if source_info.get("snowflake_uploaded"):
        return

    filename = source_info.get("file_name", f"widget_pdf_{conversation_id[:8]}.pdf")

    # Mark as upload in progress to prevent duplicate uploads
    source_info["snowflake_uploaded"] = True

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Triggering Snowflake upload for widget PDF: {filename}")

    # Create and schedule the async upload task
    async def do_upload():
        success, message, stage_path = await upload_pdf_bytes_to_snowflake(
            client, pdf_bytes, filename, conversation_id
        )
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Widget PDF upload result: {success}, {message}")
        if success and stage_path:
            source_info["stage_path"] = stage_path
            source_info["snowflake_source_type"] = "snowflake"

    # Try to get the running event loop, or create a new one
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(do_upload())
    except RuntimeError:
        # No running loop - run in a new thread
        import threading

        def run_async():
            asyncio.run(do_upload())

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()


async def handle_get_widget_data_tool_call(
    tool_args: dict,
    widgets_primary: list | None,
    widgets_secondary: list | None,
    client,
    conversation_id: str,
) -> tuple[str, dict]:
    """
    Handle get_widget_data tool calls from the LLM.

    This intercepts tool calls for get_widget_data to:
    1. Check if the document is already parsed in Snowflake (by filename)
    2. If parsed, return the content directly without fetching from widget
    3. Otherwise, look up widget by UUID and fetch data

    Args:
        tool_args: Arguments from the tool call (should contain widget_uuids)
        widgets_primary: List of primary widgets from request.widgets
        widgets_secondary: List of secondary widgets from request.widgets
        client: The Snowflake client
        conversation_id: The conversation ID

    Returns:
        Tuple of (llm_output, raw_data) like other tool executors
    """
    import asyncio
    from openbb_ai import get_widget_data
    from openbb_ai.models import WidgetRequest

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    # Extract widget UUIDs from tool args
    widget_uuids = tool_args.get("widget_uuids", [])
    if isinstance(widget_uuids, str):
        widget_uuids = [widget_uuids]

    if not widget_uuids:
        # If no specific UUIDs, use all primary widgets
        widget_uuids = []
        if widgets_primary:
            widget_uuids.extend([str(w.uuid) for w in widgets_primary])

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] handle_get_widget_data_tool_call: widget_uuids={widget_uuids}")

    # Build a lookup of available widgets
    all_widgets = []
    if widgets_primary:
        all_widgets.extend(widgets_primary)
    if widgets_secondary:
        all_widgets.extend(widgets_secondary)

    widget_lookup = {str(w.uuid): w for w in all_widgets}

    # Find requested widgets
    matched_widgets = []
    for uuid_str in widget_uuids:
        if uuid_str in widget_lookup:
            matched_widgets.append(widget_lookup[uuid_str])
        else:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Widget UUID not found: {uuid_str}")

    if not matched_widgets:
        return "No matching widgets found for the requested UUIDs.", {
            "error": "No widgets found"
        }

    # Check if any of the matched widgets reference documents already parsed in Snowflake
    # If so, we can return that content directly without fetching from widget
    if client:
        for widget in matched_widgets:
            known_filename = extract_filename_from_widget(widget)
            if known_filename:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(
                        f"[DEBUG] Checking if {known_filename} is already parsed in Snowflake"
                    )

                try:
                    # Query DOCUMENT_PARSE_RESULTS to see if file exists
                    db_name = "OPENBB_AGENTS"
                    snowflake_user = await run_in_thread(client.get_current_user)
                    sanitized_user = "".join(
                        c if c.isalnum() else "_" for c in snowflake_user
                    )
                    schema_name = f"USER_{sanitized_user}".upper()
                    qualified_table = (
                        f'"{db_name}"."{schema_name}"."DOCUMENT_PARSE_RESULTS"'
                    )

                    escaped_filename = known_filename.replace("'", "''")
                    query = f"""
                    SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
                    FROM {qualified_table}
                    WHERE FILE_NAME = '{escaped_filename}'
                    ORDER BY PAGE_NUMBER
                    """

                    result_json = await run_in_thread(client.execute_statement, query)
                    rows = json.loads(result_json) if result_json else []

                    if rows:
                        # Document is already parsed - return content directly
                        if os.environ.get("SNOWFLAKE_DEBUG"):
                            print(
                                f"[DEBUG] Document {known_filename} already parsed in Snowflake ({len(rows)} pages)"
                            )

                        # Build content from parsed pages
                        content_parts = []
                        for row in rows:
                            page_num = get_row_value(row, "PAGE_NUMBER", "page_number")
                            content = get_row_value(row, "PAGE_CONTENT", "page_content")
                            if content:
                                content_parts.append(f"[Page {page_num}]\n{content}")

                        parsed_content = "\n\n".join(content_parts)

                        # Store in snowflake_document_pages for citation matching
                        snowflake_document_pages[conversation_id] = [
                            {
                                "page": int(
                                    get_row_value(row, "PAGE_NUMBER", "page_number")
                                ),
                                "content": get_row_value(
                                    row, "PAGE_CONTENT", "page_content"
                                ),
                                "file_name": known_filename,
                            }
                            for row in rows
                        ]

                        # Store document source info
                        document_sources[conversation_id] = {
                            "source_type": "snowflake",
                            "file_name": known_filename,
                            "has_snowflake_content": True,
                            "snowflake_page_count": len(rows),
                        }

                        return (
                            f"Document '{known_filename}' content ({len(rows)} pages):\n\n{parsed_content}",
                            {
                                "source": "snowflake_cache",
                                "file_name": known_filename,
                                "page_count": len(rows),
                            },
                        )
                except Exception as e:
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Error checking Snowflake for {known_filename}: {e}"
                        )
                    # Continue to fetch from widget

    # Build widget requests
    widget_requests = []
    for widget in matched_widgets:
        input_args = {}
        if hasattr(widget, "params") and widget.params:
            input_args = {p.name: p.current_value for p in widget.params}

        widget_req = WidgetRequest(
            widget=widget,
            input_arguments=input_args,
        )
        widget_requests.append(widget_req)

    # Get widget data request object (this is what tells the UI to fetch data)
    widget_data_request = get_widget_data(widget_requests)

    # Store widget info for later processing when data comes back
    # The actual data will come back in a subsequent request as a tool message
    widget_info = {
        "widget_uuids": widget_uuids,
        "widget_names": [getattr(w, "name", str(w.uuid)) for w in matched_widgets],
        "request_sent": True,
    }

    # Return the widget data request for the UI to execute
    # This will trigger the UI to fetch the actual data
    output = f"Requesting data for {len(matched_widgets)} widget(s): {', '.join(widget_info['widget_names'])}"

    return output, {
        "widget_data_request": widget_data_request.model_dump(),
        "widget_info": widget_info,
    }


async def store_widget_data_in_snowflake(
    client,
    widget_uuid: str,
    widget_name: str,
    data_content: str | dict | list,
    conversation_id: str,
    data_type: str = "json",
) -> tuple[bool, str]:
    """
    Store widget data in Snowflake for reference and future queries.

    Args:
        client: The Snowflake client
        widget_uuid: The widget's UUID
        widget_name: The widget's name
        data_content: The data content (JSON, text, etc.)
        conversation_id: The conversation ID
        data_type: Type of data ("json", "pdf", "text")

    Returns:
        Tuple of (success, message)
    """
    import asyncio

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        # Get user-specific schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()

        qualified_table = f"{db_name}.{schema_name}.WIDGET_DATA_CACHE"

        # Create table if not exists
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {qualified_table} (
            WIDGET_UUID STRING,
            WIDGET_NAME STRING,
            CONVERSATION_ID STRING,
            DATA_TYPE STRING,
            DATA_CONTENT VARIANT,
            CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
            UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        await run_in_thread(client.execute_statement, create_table_query)

        # Serialize data content
        if isinstance(data_content, (dict, list)):
            data_json = json.dumps(data_content)
        else:
            data_json = json.dumps({"content": str(data_content)})

        # Escape for SQL
        escaped_uuid = widget_uuid.replace("'", "''")
        escaped_name = widget_name.replace("'", "''")
        escaped_conv_id = conversation_id.replace("'", "''")
        escaped_data = data_json.replace("'", "''")

        # Upsert (merge) the data
        merge_query = f"""
        MERGE INTO {qualified_table} AS target
        USING (SELECT '{escaped_uuid}' AS uuid, '{escaped_conv_id}' AS conv_id) AS source
        ON target.WIDGET_UUID = source.uuid AND target.CONVERSATION_ID = source.conv_id
        WHEN MATCHED THEN
            UPDATE SET DATA_CONTENT = PARSE_JSON('{escaped_data}'), 
                       DATA_TYPE = '{data_type}',
                       UPDATED_AT = CURRENT_TIMESTAMP()
        WHEN NOT MATCHED THEN
            INSERT (WIDGET_UUID, WIDGET_NAME, CONVERSATION_ID, DATA_TYPE, DATA_CONTENT)
            VALUES ('{escaped_uuid}', '{escaped_name}', '{escaped_conv_id}', '{data_type}', PARSE_JSON('{escaped_data}'))
        """
        await run_in_thread(client.execute_statement, merge_query)

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(
                f"[DEBUG] Stored widget data for {widget_name} ({widget_uuid}) in Snowflake"
            )

        return True, "Widget data stored in Snowflake"

    except Exception as e:
        error_msg = f"Failed to store widget data: {e}"
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] {error_msg}")
        return False, error_msg


async def check_existing_snowflake_document(
    client,
    pdf_bytes: bytes,
    conversation_id: str,
    known_filename: str | None = None,
) -> dict | None:
    """
    Check if a PDF document already exists in Snowflake DOCUMENT_PARSE_RESULTS.

    Uses the known filename if provided, otherwise falls back to hash-based lookup.
    If the document exists and has been fully parsed, returns the parsed content and page count.

    Args:
        client: The Snowflake client
        pdf_bytes: The PDF file bytes to check
        conversation_id: The conversation ID
        known_filename: Optional filename from widget metadata (preferred over hash)

    Returns:
        dict with parsed content info if exists and complete, None otherwise.
        Format: {
            "exists": True,
            "file_name": str,
            "page_count": int,
            "pages": list of {"page": int, "content": str},
            "stage_path": str
        }
    """
    import asyncio

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        # Use known filename if provided, otherwise fall back to hash-based name
        if known_filename:
            file_name = known_filename
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Checking Snowflake for known filename: {file_name}")
        else:
            # Create a hash of the PDF to use as identifier
            pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
            file_name = f"widget_pdf_{pdf_hash}.pdf"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Checking Snowflake for hash-based filename: {file_name}"
                )

        # Get user-specific schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()

        qualified_table = f'"{db_name}"."{schema_name}"."DOCUMENT_PARSE_RESULTS"'

        # Check if document exists in DOCUMENT_PARSE_RESULTS
        escaped_filename = file_name.replace("'", "''")
        check_query = f"""
        SELECT FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, PARSED_AT
        FROM {qualified_table}
        WHERE FILE_NAME = '{escaped_filename}'
        ORDER BY PAGE_NUMBER
        """

        result_json = await run_in_thread(client.execute_statement, check_query)
        rows = json.loads(result_json) if result_json else []

        if not rows:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Document {file_name} not found in Snowflake")
            return None

        # Document exists - extract pages
        pages = []
        stage_path = None
        for row in rows:
            page_num = get_row_value(row, "PAGE_NUMBER", "page_number")
            content = get_row_value(row, "PAGE_CONTENT", "page_content")
            if not stage_path:
                stage_path = get_row_value(row, "STAGE_PATH", "stage_path")

            if page_num and content:
                pages.append({"page": int(page_num), "content": content})

        if pages:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Found existing document {file_name} with {len(pages)} pages in Snowflake"
                )

            return {
                "exists": True,
                "file_name": file_name,
                "page_count": len(pages),
                "pages": pages,
                "stage_path": stage_path,
                "pdf_hash": pdf_hash,
            }

        return None

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Error checking existing Snowflake document: {e}")
        return None


def check_existing_snowflake_document_sync(
    client,
    pdf_bytes: bytes,
    conversation_id: str,
    known_filename: str | None = None,
) -> dict | None:
    """Synchronous wrapper for check_existing_snowflake_document."""
    import asyncio

    try:
        loop = asyncio.get_running_loop()
        # Already in async context - use run_coroutine_threadsafe
        import concurrent.futures

        future = asyncio.run_coroutine_threadsafe(
            check_existing_snowflake_document(
                client, pdf_bytes, conversation_id, known_filename
            ),
            loop,
        )
        return future.result(timeout=10)
    except RuntimeError:
        # No running loop - create new one
        return asyncio.run(
            check_existing_snowflake_document(
                client, pdf_bytes, conversation_id, known_filename
            )
        )
    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Error in sync wrapper: {e}")
        return None


async def load_snowflake_document_pages(
    client,
    conversation_id: str,
    file_name: str | None = None,
) -> bool:
    """
    Load document pages from DOCUMENT_PARSE_RESULTS for citation matching.

    Args:
        client: The Snowflake client
        conversation_id: The conversation ID
        file_name: Optional specific file to load. If None, loads the most recent.

    Returns:
        True if pages were loaded, False otherwise.
    """
    import asyncio

    async def run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    try:
        # Get user-specific schema
        db_name = "OPENBB_AGENTS"
        snowflake_user = await run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        schema_name = f"USER_{sanitized_user}".upper()

        qualified_table = f"{db_name}.{schema_name}.DOCUMENT_PARSE_RESULTS"

        # Build query
        if file_name:
            # Escape single quotes in filename
            escaped_filename = file_name.replace("'", "''")
            query = f"""
            SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT 
            FROM {qualified_table} 
            WHERE FILE_NAME = '{escaped_filename}'
            ORDER BY PAGE_NUMBER
            """
        else:
            # Get most recent document
            query = f"""
            SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT 
            FROM {qualified_table} 
            WHERE PARSED_AT = (SELECT MAX(PARSED_AT) FROM {qualified_table})
            ORDER BY PAGE_NUMBER
            """

        result_json = await run_in_thread(client.execute_statement, query)
        rows = json.loads(result_json) if result_json else []

        if not rows:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] No document pages found for {file_name or 'most recent'}"
                )
            return False

        # Store pages for citation matching
        pages = []
        detected_file_name = None
        for row in rows:
            fn = get_row_value(row, "FILE_NAME", "file_name")
            page_num = get_row_value(row, "PAGE_NUMBER", "page_number")
            content = get_row_value(row, "PAGE_CONTENT", "page_content")

            if fn and not detected_file_name:
                detected_file_name = fn

            if page_num and content:
                pages.append(
                    {
                        "page": int(page_num),
                        "content": content,
                        "file_name": fn or file_name or "unknown",
                    }
                )

        if pages:
            snowflake_document_pages[conversation_id] = pages

            # Update document_sources
            document_sources[conversation_id] = {
                "source_type": "snowflake",
                "file_name": detected_file_name or file_name,
                "has_snowflake_pages": True,
            }

            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Loaded {len(pages)} pages from Snowflake for {detected_file_name or file_name}"
                )
            return True

        return False

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Failed to load Snowflake document pages: {e}")
        return False


def find_best_match_in_snowflake_pages(
    query_text: str,
    conversation_id: str,
    preferred_page: int | None = None,
) -> dict | None:
    """Find the best matching content in Snowflake document pages.

    Returns a dict with page, content, x0, top, x1, bottom for citation.
    """
    from difflib import SequenceMatcher

    if conversation_id not in snowflake_document_pages:
        return None

    pages = snowflake_document_pages[conversation_id]
    if not pages:
        return None

    query_lower = query_text.lower().strip()

    # Extract meaningful words
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    query_words = [
        word
        for word in re.findall(r"\b[a-z]+\b", query_lower)
        if word not in stopwords and len(word) > 2
    ]

    if not query_words:
        return None

    best_match = None
    best_score = 0

    for page_data in pages:
        content_lower = page_data["content"].lower()

        # Count word matches
        matching_words = sum(1 for word in query_words if word in content_lower)

        # Sequence similarity
        seq_matcher = SequenceMatcher(None, query_lower, content_lower[:500])
        similarity_ratio = seq_matcher.ratio()

        score = (matching_words * 10) + (similarity_ratio * 100)

        if preferred_page and page_data["page"] == preferred_page:
            score *= 1.5

        if score > best_score:
            best_score = score
            best_match = page_data

    if best_score < 30:
        return None

    if best_match:
        # Create bounding box approximation (full page width, estimated position)
        # Since Cortex doesn't give us exact coordinates, we use page-level reference
        return {
            "text": (
                best_match["content"][:200] + "..."
                if len(best_match["content"]) > 200
                else best_match["content"]
            ),
            "page": best_match["page"],
            "x0": 50,  # Approximate left margin
            "top": 100,  # Approximate top position
            "x1": 550,  # Approximate right margin
            "bottom": 200,  # Approximate bottom position
            "source": "snowflake",
            "file_name": best_match["file_name"],
        }

    return None


def parse_widget_data(
    data: DataContent | DataFileReferences | list | dict | str,
    conversation_id: str,
    client=None,
    known_filename: str | None = None,
) -> str:
    """Parse widget data into a readable string format.

    Args:
        data: The widget data to parse
        conversation_id: The conversation ID for tracking
        client: Optional Snowflake client for checking existing parsed PDFs
        known_filename: Optional filename from widget metadata (instead of hash-based name)
    """
    result_parts = []

    # If it's a string, just return it
    if isinstance(data, str):
        return data

    # Handle dict that might contain widget results
    if isinstance(data, dict):
        # Check if it's a dict of widget UUIDs to results
        for widget_id, widget_result in data.items():
            result_parts.append(f"### Widget: {widget_id}\n")
            if isinstance(widget_result, str):
                result_parts.append(widget_result)
            elif isinstance(widget_result, dict):
                result_parts.append(json.dumps(widget_result, indent=2))
            elif isinstance(widget_result, list):
                for item in widget_result:
                    if hasattr(item, "data_format"):
                        result_parts.append(
                            parse_single_data_item(
                                item, conversation_id, client, known_filename
                            )
                        )
                    else:
                        result_parts.append(str(item))
            else:
                result_parts.append(str(widget_result))
            result_parts.append("\n")
        return "\n".join(result_parts)

    # Handle list - this is the main case we're hitting
    if isinstance(data, list):
        for idx, item in enumerate(data):
            result_parts.append(f"### Data Source {idx + 1}\n")

            # Check if it's a Pydantic model with 'items' attribute (DataContent or DataFileReferences)
            if hasattr(item, "items"):
                items_list = getattr(item, "items", [])
                if isinstance(items_list, list):
                    for result_item in items_list:
                        if hasattr(result_item, "data_format"):
                            result_parts.append(
                                parse_single_data_item(
                                    result_item, conversation_id, client, known_filename
                                )
                            )
                        # Try to get content directly
                        elif hasattr(result_item, "content"):
                            content = getattr(result_item, "content")
                            if isinstance(content, str):
                                result_parts.append(content)
                            elif isinstance(content, (dict, list)):
                                result_parts.append(json.dumps(content, indent=2))
                            else:
                                result_parts.append(str(content))
                        else:
                            result_parts.append(str(result_item))
                else:
                    result_parts.append(str(items_list))

            # Check if item has data_format directly
            elif hasattr(item, "data_format"):
                result_parts.append(
                    parse_single_data_item(
                        item, conversation_id, client, known_filename
                    )
                )

            # Check if it's a dict
            elif isinstance(item, dict):
                result_parts.append(json.dumps(item, indent=2))

            # Check if it has content attribute directly
            elif hasattr(item, "content"):
                content = getattr(item, "content")
                if isinstance(content, str):
                    result_parts.append(content)
                elif isinstance(content, (dict, list)):
                    result_parts.append(json.dumps(content, indent=2))
                else:
                    result_parts.append(str(content))

            # Fallback - try to convert to dict if it's a Pydantic model
            elif hasattr(item, "model_dump"):
                try:
                    dumped = item.model_dump()
                    result_parts.append(json.dumps(dumped, indent=2))
                except Exception:
                    result_parts.append(str(item))

            # Last resort
            else:
                result_parts.append(str(item))

            result_parts.append("\n")

        return "\n".join(result_parts)

    # Single item - not a list or dict
    # Check if item has data_format attribute (SingleDataContent or SingleFileReference)
    if hasattr(data, "data_format"):
        return parse_single_data_item(data, conversation_id, client, known_filename)

    # Handle items attribute (DataContent or DataFileReferences)
    if hasattr(data, "items"):
        items_list = getattr(data, "items", [])
        if isinstance(items_list, list):
            for result in items_list:
                result_parts.append(
                    parse_single_data_item(
                        result, conversation_id, client, known_filename
                    )
                )
            return "\n\n".join(result_parts)

    # Try model_dump for Pydantic models
    if hasattr(data, "model_dump"):
        try:
            dumped = data.model_dump()
            return json.dumps(dumped, indent=2)
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(e)  # noqa: T201

    # Fallback
    return str(data)


def extract_pdf_with_positions(pdf_bytes: bytes) -> Tuple[str, list[dict]]:
    """Extract text and character positions from PDF using pdfplumber.

    Returns:
        Tuple of (full_text, text_positions)
        where text_positions contains dicts with text, page, x0, top, x1, bottom
    """
    import pdfplumber

    document_text = ""
    text_positions = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Use extract_words for reliable text extraction with coordinates
            # This avoids the issue of mashed characters when using page.chars directly
            words = page.extract_words(
                keep_blank_chars=False, x_tolerance=3, y_tolerance=3
            )

            # Group words into lines based on 'top' coordinate
            lines = {}
            for word in words:
                # Rounding top to nearest integer helps group words on same line
                top = round(word["top"])
                if top not in lines:
                    lines[top] = []
                lines[top].append(word)

            # Sort lines by vertical position
            sorted_tops = sorted(lines.keys())

            for top in sorted_tops:
                line_words = lines[top]
                # Sort words in line by horizontal position
                line_words.sort(key=lambda w: w["x0"])

                # Reconstruct line text
                line_text = " ".join(w["text"] for w in line_words)

                if len(line_text.strip()) > 5:  # Filter out noise
                    # Calculate bounding box for the line
                    x0 = min(w["x0"] for w in line_words)
                    x1 = max(w["x1"] for w in line_words)
                    bottom = max(w["bottom"] for w in line_words)

                    text_positions.append(
                        {
                            "text": line_text,
                            "page": page_num,
                            "x0": x0,
                            "top": top,
                            "x1": x1,
                            "bottom": bottom,
                        }
                    )

            # Extract full text for context
            page_text = page.extract_text()
            if page_text:
                document_text += page_text + "\n\n"

    return document_text, text_positions


def find_best_match(
    query_text: str,
    pdf_positions: list[dict],
    preferred_page: int | None = None,
) -> dict | None:
    """Find the best matching line in PDF positions for the query text.

    This function finds the PDF line that best matches what the LLM just said.
    """
    import re
    from difflib import SequenceMatcher

    if not query_text or not pdf_positions:
        return None

    query_lower = query_text.lower().strip()

    # Extract ALL meaningful words from the query (what the LLM just wrote)
    # Don't hardcode specific terms - extract what's ACTUALLY in the text!
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    # Get actual words from what the LLM wrote
    query_words = [
        word
        for word in re.findall(r"\b[a-z]+\b", query_lower)
        if word not in stopwords and len(word) > 2
    ]

    if not query_words:
        return None

    best_match = None
    best_score = 0

    # Check each PDF line for similarity to what the LLM just wrote
    for position in pdf_positions:
        line_lower = position["text"].lower()

        # Calculate how many of the LLM's words appear in this PDF line
        matching_words = sum(1 for word in query_words if word in line_lower)

        # Use sequence matching to find similar phrases
        seq_matcher = SequenceMatcher(None, query_lower, line_lower)
        similarity_ratio = seq_matcher.ratio()

        # Combined score: word matches + sequence similarity
        score = (matching_words * 10) + (similarity_ratio * 100)

        if preferred_page and position.get("page") == preferred_page:
            score *= 1.5  # bias toward hinted page
        if score > best_score:
            best_score = score
            best_match = position

    # Only return a match if it's actually relevant (not just random overlap)
    min_threshold = 30  # Require decent score to avoid random matches

    if best_score < min_threshold:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] No good match found for citation. Best score: {best_score}")
            print(f"[DEBUG] Query words: {query_words[:10]}")
        return None

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] Found match with score {best_score}: {best_match['text'][:80]}..."
        )

    return best_match


def parse_single_data_item(
    result, conversation_id: str, client=None, known_filename: str | None = None
) -> str:
    """Parse a single data item (SingleDataContent or SingleFileReference).

    When a PDF is received from widget data:
    1. Check if the file already exists in Snowflake DOCUMENT_PARSE_RESULTS
    2. If parsing is complete, use the Snowflake-parsed content for the LLM
    3. Always keep widget PDF bytes/positions for citation highlighting in the UI

    Args:
        result: The data item to parse
        conversation_id: The conversation ID for tracking
        client: Optional Snowflake client for checking/uploading PDFs to Snowflake
        known_filename: Optional filename from widget metadata (instead of hash-based name)
    """
    data_format = result.data_format

    # Parse PDF content
    if isinstance(data_format, PdfDataFormat):
        if isinstance(result, SingleDataContent):
            try:
                # Decode base64 to bytes
                pdf_bytes = base64.b64decode(result.content)

                # Always extract text with positions using pdfplumber for citation highlighting
                # This is the source for highlighting in the UI
                full_text, text_positions = extract_pdf_with_positions(pdf_bytes)

                # Store text positions for citation generation (always from widget PDF)
                pdf_text_blocks[conversation_id] = text_positions

                # Use known filename if provided, otherwise fall back to hash-based name
                if known_filename:
                    widget_file_name = known_filename
                    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Using known filename: {widget_file_name} (hash: {pdf_hash})"
                        )
                else:
                    # Create hash-based filename for consistent identification
                    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
                    widget_file_name = f"widget_pdf_{pdf_hash}.pdf"

                # Initialize document source with widget info (for UI citation highlighting)
                document_sources[conversation_id] = {
                    "source_type": "widget",
                    "file_name": widget_file_name,
                    "pdf_bytes": pdf_bytes,
                    "has_pdfplumber_positions": True,
                    "pdf_hash": pdf_hash,
                }

                # Check if this PDF already exists in Snowflake with completed parsing
                snowflake_content = None
                if client:
                    try:
                        existing_doc = check_existing_snowflake_document_sync(
                            client, pdf_bytes, conversation_id, known_filename
                        )
                        if existing_doc and existing_doc.get("exists"):
                            # Document already parsed in Snowflake - use that content for LLM
                            pages = existing_doc.get("pages", [])
                            if pages:
                                # Build content from Snowflake-parsed pages
                                snowflake_content = "\n\n".join(
                                    f"[Page {p['page']}]\n{p['content']}" for p in pages
                                )

                                # Store Snowflake pages for additional citation matching
                                snowflake_document_pages[conversation_id] = [
                                    {
                                        "page": p["page"],
                                        "content": p["content"],
                                        "file_name": existing_doc["file_name"],
                                    }
                                    for p in pages
                                ]

                                # Update document source to indicate Snowflake content is available
                                # But keep widget info for citation highlighting
                                document_sources[conversation_id].update(
                                    {
                                        "has_snowflake_content": True,
                                        "snowflake_file_name": existing_doc[
                                            "file_name"
                                        ],
                                        "snowflake_page_count": existing_doc[
                                            "page_count"
                                        ],
                                        "stage_path": existing_doc.get("stage_path"),
                                    }
                                )

                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(
                                        f"[DEBUG] Using Snowflake-parsed content ({len(pages)} pages) "
                                        f"for LLM, widget PDF for citations"
                                    )
                    except Exception as e:
                        if os.environ.get("SNOWFLAKE_DEBUG"):
                            print(
                                f"[DEBUG] Snowflake check failed, using pdfplumber: {e}"
                            )

                # Use Snowflake content if available, otherwise pdfplumber extraction
                content_for_llm = snowflake_content if snowflake_content else full_text

                return f"[PDF Content]:\n{content_for_llm}"
            except Exception as e:
                # Fallback to original decoding if pdfplumber fails
                try:
                    content = base64.b64decode(result.content).decode(
                        "utf-8", errors="ignore"
                    )
                    return f"[PDF Content]:\n{content}"
                except Exception as fallback_e:
                    return f"[PDF Content - Unable to parse: {e}, fallback failed: {fallback_e}]"
        elif isinstance(result, SingleFileReference):
            return f"[PDF File]: {result.url}"

    # Parse Image data
    elif isinstance(data_format, ImageDataFormat):
        if isinstance(result, SingleDataContent):
            return f"[Image Data - Base64 encoded, {len(result.content)} bytes]"
        if isinstance(result, SingleFileReference):
            return f"[Image File]: {result.url}"

    # Parse Spreadsheet data
    elif isinstance(data_format, SpreadsheetDataFormat):
        if isinstance(result, SingleDataContent):
            return f"[Spreadsheet Data]:\n{result.content}"
        if isinstance(result, SingleFileReference):
            return f"[Spreadsheet File]: {result.url}"

    # Parse RawObject data (JSON/dict)
    elif isinstance(data_format, RawObjectDataFormat):
        if isinstance(result, SingleDataContent):
            try:
                if isinstance(result.content, str):
                    parsed = json.loads(result.content)
                else:
                    parsed = result.content
                formatted = json.dumps(parsed, indent=2)
                return f"[Data Object]:\n{formatted}"
            except Exception:
                return f"[Data Object]:\n{result.content}"
        if isinstance(result, SingleFileReference):
            return f"[Data File]: {result.url}"

    # Fallback for unknown formats
    elif isinstance(result, SingleDataContent):
        return f"[Unknown Format Data]: {str(result.content)[:500]}"
    elif isinstance(result, SingleFileReference):
        return f"[Unknown Format File]: {result.url}"

    return str(result)


def cleanup_text(text: str) -> str:
    """Clean up text from the LLM response."""
    text = text.strip()
    # Remove wrapping quotes
    while (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    # Replace escape sequences
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "")
    text = text.replace('\\"', '"')
    text = text.replace("\\-", "-")

    # Fix bullet points
    text = text.replace("\t-", "-")
    text = text.replace("\t", "")

    # Ensure bullet points are formatted properly
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        new_line = line.strip()
        if new_line.startswith("-") and not new_line.startswith("- "):
            new_line = "- " + new_line[1:].strip()
        formatted_lines.append(new_line)

    return "\n".join(formatted_lines)


def cleanup_identifier(identifier: str) -> str:
    """
    Cleans up a Snowflake identifier (like a table name) by removing common
    LLM-added artifacts like quotes, backticks, and leading/trailing whitespace.
    """
    if not identifier:
        return ""

    # Remove leading/trailing whitespace
    cleaned = identifier.strip()

    # Remove surrounding quotes or backticks
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (
        cleaned.startswith('"') and cleaned.endswith('"')
    ):
        cleaned = cleaned[1:-1]

    # A second strip to handle cases like "' table_name '"
    cleaned = cleaned.strip()

    # The LLM sometimes returns markdown-style backticks, remove them
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1]

    # Remove any remaining backslashes which might escape quotes
    cleaned = cleaned.replace("\\", "")

    return cleaned


def to_sse(sse_event):
    """Convert the SSE event model to a dictionary for EventSourceResponse."""
    return {"event": sse_event.event, "data": json.dumps(sse_event.data.model_dump())}


def find_quote_in_pdf_blocks(quote: str, conversation_id: str) -> dict | None:
    """Find a quote in the PDF text blocks and return its position data."""
    if conversation_id not in pdf_text_blocks:
        return None

    pdf_positions = pdf_text_blocks[conversation_id]
    quote_lower = quote.lower().strip()

    # Try exact match first
    for position in pdf_positions:
        if quote_lower in position["text"].lower():
            return position

    # Try partial match (first 50 chars)
    if len(quote) > 50:
        quote_start = quote_lower[:50]
        for position in pdf_positions:
            if quote_start in position["text"].lower():
                return position

    # Try key words match
    quote_words = quote_lower.split()
    if len(quote_words) > 5:
        key_words = quote_words[:5]  # First 5 words
        for position in pdf_positions:
            text_lower = position["text"].lower()
            if all(word in text_lower for word in key_words):
                return position

    return None


def extract_quotes_from_llm_response(response_text: str) -> list[tuple[str, int]]:
    """Extract citation references from LLM response.

    Returns list of (surrounding_text, citation_number) tuples.
    """
    import re

    citations = []

    # Find all [N] citation references
    pattern = r"([^.!?]*?)\[(\d+)\]"
    matches = re.findall(pattern, response_text)

    for context_text, citation_num in matches:
        # Get the sentence containing this citation
        sentence = context_text.strip()
        if len(sentence) > 200:
            sentence = sentence[-200:]  # Last 200 chars before citation
        citations.append((sentence, int(citation_num)))

    # If no numbered citations found, try old format as fallback
    if not citations:
        # Pattern for "Quote text" (Page X) format
        pattern_old = r'"([^"]+)"\s*\((?:[Pp]age|[Pp]\.?)\s*(\d+)\)'
        matches_old = re.findall(pattern_old, response_text)
        for idx, (quote, page) in enumerate(matches_old, 1):
            citations.append((quote, idx))  # Use index as citation number

    return citations


def create_citations_from_widget_data(
    widget_data: list,
    widgets_primary: list,
    input_arguments: dict,
    specific_citation_num: int = None,
):
    """Create citations with PDF highlighting from widget data or Snowflake documents.

    This function now supports both:
    1. Widget PDFs (extracted via pdfplumber with precise positions)
    2. Snowflake documents (stored in DOCUMENT_PARSE_RESULTS with page-level content)
    """
    from openbb_ai import cite
    from openbb_ai.models import CitationHighlightBoundingBox

    citations_list = []
    conversation_id = input_arguments.get("conversation_id", "default")

    if not widget_data or not widgets_primary:
        return citations_list

    # Get the widget that contains the PDF
    widget = widgets_primary[0] if widgets_primary else None
    if not widget:
        return citations_list

    # Determine the citation source (widget PDF positions or Snowflake pages)
    source_info = document_sources.get(conversation_id, {})
    source_type = source_info.get("source_type", "widget")

    # Check for available position data
    has_pdfplumber_positions = (
        conversation_id in pdf_text_blocks and pdf_text_blocks[conversation_id]
    )
    has_snowflake_pages = (
        conversation_id in snowflake_document_pages
        and snowflake_document_pages[conversation_id]
    )

    if not has_pdfplumber_positions and not has_snowflake_pages:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] No position data for conversation {conversation_id}")
        return citations_list

    # For specific citation number, find what the LLM is ACTUALLY citing
    if specific_citation_num is not None:
        selected_position = None

        # Check what the LLM was referencing when it added [N]
        if conversation_id in llm_referenced_quotes:
            # Find the context for THIS citation number
            for context_text, cite_num in llm_referenced_quotes[conversation_id]:
                if cite_num == specific_citation_num:
                    # The LLM was talking about context_text when it added [N]
                    # Find the BEST matching position
                    import re

                    # Get the last sentence(s)
                    sentences = re.split(r"(?<=[.!?])\s+", context_text)
                    sentences = [s for s in sentences if s.strip()]

                    if sentences:
                        query = sentences[-1]
                        # If last sentence is very short, take previous too
                        if len(query) < 50 and len(sentences) > 1:
                            query = sentences[-2] + " " + query

                        # Try pdfplumber positions first (more precise)
                        if has_pdfplumber_positions:
                            pdf_positions = pdf_text_blocks[conversation_id]
                            selected_position = find_best_match(query, pdf_positions)
                            if selected_position:
                                selected_position["source"] = "pdfplumber"

                        # Fall back to Snowflake pages if no pdfplumber match
                        if not selected_position and has_snowflake_pages:
                            selected_position = find_best_match_in_snowflake_pages(
                                query, conversation_id
                            )

                        if selected_position and os.environ.get("SNOWFLAKE_DEBUG"):
                            print(
                                f"[DEBUG] Matched citation [{specific_citation_num}] to page {selected_position['page']} "
                                f"(source: {selected_position.get('source', 'unknown')})"
                            )
                    break

        # Create the citation object
        extra_details = {
            "Page": selected_position["page"] if selected_position else 1,
            "Reference": (
                selected_position["text"][:100] + "..."
                if selected_position and len(selected_position["text"]) > 100
                else (
                    selected_position["text"]
                    if selected_position
                    else "Document Reference"
                )
            ),
        }

        # Add source indicator if from Snowflake
        if selected_position and selected_position.get("source") == "snowflake":
            if selected_position.get("file_name"):
                extra_details["Document"] = selected_position["file_name"]

        citation_obj = cite(
            widget=widget,
            input_arguments=input_arguments,
            extra_details=extra_details,
        )

        # Only add bounding box if we actually found a match
        if selected_position:
            citation_obj.quote_bounding_boxes = [
                [
                    CitationHighlightBoundingBox(
                        text=selected_position["text"],
                        page=selected_position["page"],
                        x0=selected_position["x0"],
                        top=selected_position["top"],
                        x1=selected_position["x1"],
                        bottom=selected_position["bottom"],
                    )
                ]
            ]

        citations_list.append(citation_obj)

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(
                f"[DEBUG] Created citation [{specific_citation_num}] -> "
                f"widget: {widget.uuid if hasattr(widget, 'uuid') else 'unknown'}, "
                f"match found: {selected_position is not None}, "
                f"source: {selected_position.get('source') if selected_position else 'none'}"
            )

    return citations_list
