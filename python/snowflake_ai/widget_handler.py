"""Generic widget data handler for Snowflake AI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from typing import Any

from openbb_ai.models import (
    ImageDataFormat,
    PdfDataFormat,
    RawObjectDataFormat,
    SingleDataContent,
    SingleFileReference,
    SpreadsheetDataFormat,
)
import pdfplumber

from .document_processor import DocumentProcessor
from .logger import get_logger

logger = get_logger(__name__)


class WidgetHandler:
    """Handles generic widget data processing and conversation-scoped caching."""

    _instance: "WidgetHandler | None" = None
    _instance_lock = asyncio.Lock()

    def __init__(self) -> None:
        pass

    @classmethod
    async def instance(cls) -> "WidgetHandler":
        async with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @staticmethod
    async def _run_in_thread(func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    @staticmethod
    async def _get_user_schema(client) -> str:
        snowflake_user = await WidgetHandler._run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        return f"USER_{sanitized_user}".upper()

    @staticmethod
    def _normalize_params(params: dict[str, Any] | None) -> str:
        if not params:
            return "{}"
        return json.dumps(params, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def is_document_widget(widget) -> bool:
        if not hasattr(widget, "params"):
            return False

        for param in widget.params:
            current_value = getattr(param, "current_value", None)
            multi_select = getattr(param, "multi_select", False)

            if multi_select and isinstance(current_value, list):
                for value in current_value:
                    if not isinstance(value, str):
                        continue

                    value_lower = value.lower()

                    if value.startswith("@") and any(
                        ext in value_lower for ext in [".pdf", ".txt", ".doc", ".docx"]
                    ):
                        return True

                    if value.startswith(("http://", "https://")) and any(
                        ext in value_lower for ext in [".pdf", ".txt", ".doc", ".docx"]
                    ):
                        return True

        return False

    @staticmethod
    def extract_document_paths(widget) -> list[str]:
        if not hasattr(widget, "params"):
            return []

        paths = []
        for param in widget.params:
            current_value = getattr(param, "current_value", None)
            multi_select = getattr(param, "multi_select", False)

            if multi_select and isinstance(current_value, list):
                for value in current_value:
                    if not isinstance(value, str):
                        continue

                    value_lower = value.lower()

                    if value.startswith("@") and any(
                        ext in value_lower for ext in [".pdf", ".txt", ".doc", ".docx"]
                    ):
                        paths.append(value)
                    elif value.startswith(("http://", "https://")) and any(
                        ext in value_lower for ext in [".pdf", ".txt", ".doc", ".docx"]
                    ):
                        paths.append(value)

        return paths

    async def _ensure_widget_cache_table(self, client, user_schema: str) -> None:
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS OPENBB_AGENTS.{user_schema}.WIDGET_DATA_CACHE (
            CONVERSATION_ID VARCHAR(16777216),
            WIDGET_UUID VARCHAR(16777216),
            WIDGET_NAME VARCHAR(16777216),
            WIDGET_TYPE VARCHAR(16777216),
            INPUT_PARAMS VARIANT,
            DATA_CONTENT VARIANT,
            CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            LAST_ACCESSED TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        try:
            await self._run_in_thread(client.execute_statement, create_table_sql)
        except Exception:
            pass

    async def get_cached_widget_data(
        self,
        client,
        conversation_id: str,
        widget_uuid: str,
        current_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        user_schema = await self._get_user_schema(client)
        await self._ensure_widget_cache_table(client, user_schema)

        query = f"""
        SELECT INPUT_PARAMS, DATA_CONTENT, CREATED_AT
        FROM OPENBB_AGENTS.{user_schema}.WIDGET_DATA_CACHE
        WHERE CONVERSATION_ID = '{conversation_id.replace("'", "''")}'
          AND WIDGET_UUID = '{widget_uuid.replace("'", "''")}'
        ORDER BY CREATED_AT DESC
        LIMIT 1
        """

        try:
            result = await self._run_in_thread(client.execute_statement, query)
            if result:
                result_data = json.loads(result) if isinstance(result, str) else result
                if result_data.get("rowData"):
                    row = result_data["rowData"][0]
                    cached_params = row.get("INPUT_PARAMS") or row.get("input_params")
                    cached_data = row.get("DATA_CONTENT") or row.get("data_content")

                    if current_params is not None:
                        normalized_current = self._normalize_params(current_params)
                        normalized_cached = self._normalize_params(cached_params)

                        if normalized_current != normalized_cached:
                            return None

                    update_query = f"""
                    UPDATE OPENBB_AGENTS.{user_schema}.WIDGET_DATA_CACHE
                    SET LAST_ACCESSED = CURRENT_TIMESTAMP()
                    WHERE CONVERSATION_ID = '{conversation_id.replace("'", "''")}'
                      AND WIDGET_UUID = '{widget_uuid.replace("'", "''")}'
                    """
                    await self._run_in_thread(client.execute_statement, update_query)

                    return cached_data

        except Exception:
            pass

        return None

    async def store_widget_data(
        self,
        client,
        conversation_id: str,
        widget_uuid: str,
        widget_name: str,
        widget_type: str,
        input_params: dict[str, Any] | None,
        data_content: Any,
    ) -> None:
        user_schema = await self._get_user_schema(client)
        await self._ensure_widget_cache_table(client, user_schema)

        params_json = json.dumps(input_params) if input_params else "NULL"
        data_json = json.dumps(data_content) if data_content else "NULL"

        insert_query = f"""
        INSERT INTO OPENBB_AGENTS.{user_schema}.WIDGET_DATA_CACHE
        (CONVERSATION_ID, WIDGET_UUID, WIDGET_NAME, WIDGET_TYPE, INPUT_PARAMS, DATA_CONTENT)
        VALUES (
            '{conversation_id.replace("'", "''")}',
            '{widget_uuid.replace("'", "''")}',
            '{widget_name.replace("'", "''")}',
            '{widget_type.replace("'", "''")}',
            PARSE_JSON('{params_json.replace("'", "''")}'),
            PARSE_JSON('{data_json.replace("'", "''")}')
        )
        """

        try:
            await self._run_in_thread(client.execute_statement, insert_query)
        except Exception:
            pass

    def parse_widget_data(
        self, data, conversation_id, client=None, known_filename=None
    ) -> str:
        """Parse widget data response into a readable string format.

        Handles various data formats including PDF extraction with positions.
        """
        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            parts = []
            for widget_uuid, result_obj in data.items():
                if hasattr(result_obj, "data"):
                    parsed = self.parse_widget_data(
                        result_obj.data, conversation_id, client, known_filename
                    )
                    parts.append(f"[Widget {widget_uuid}]:\n{parsed}")
                else:
                    parts.append(f"[Widget {widget_uuid}]: {result_obj}")
            return "\n\n".join(parts)

        if isinstance(data, list):
            parsed_items = []
            for item in data:
                parsed_item = self.parse_single_data_item(
                    item, conversation_id, client, known_filename
                )
                if parsed_item:
                    parsed_items.append(parsed_item)
            return "\n\n".join(parsed_items) if parsed_items else "[No data]"

        if hasattr(data, "data_format"):
            return self.parse_single_data_item(
                data, conversation_id, client, known_filename
            )

        if hasattr(data, "model_dump"):
            dumped = data.model_dump()
            return json.dumps(dumped, indent=2)

        return str(data)

    def parse_single_data_item(
        self, item, conversation_id: str, client=None, known_filename: str | None = None
    ) -> str:
        """Parse a single data item with a data_format."""
        if not hasattr(item, "data_format"):
            return str(item)

        data_format = item.data_format
        result = item.result

        # Parse PDF data
        if isinstance(data_format, PdfDataFormat):
            if isinstance(result, SingleDataContent):
                try:
                    pdf_bytes = base64.b64decode(result.content)
                    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()

                    # For document widgets, store PDF info for citation matching
                    doc_proc = DocumentProcessor.instance()
                    doc_source = doc_proc.get_document_source(conversation_id) or {}
                    doc_source.update(
                        {
                            "widget_pdf_bytes": pdf_bytes,
                            "widget_pdf_hash": pdf_hash,
                        }
                    )
                    doc_proc.set_document_source(conversation_id, doc_source)

                    # Extract text with positions from widget PDF
                    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                        positions = []
                        full_text_parts = []

                        for page_num, page in enumerate(pdf.pages, start=1):
                            page_text = page.extract_text() or ""
                            full_text_parts.append(f"[Page {page_num}]\n{page_text}")

                            # Extract word bounding boxes for citation matching
                            words = page.extract_words(
                                x_tolerance=2, y_tolerance=2, keep_blank_chars=False
                            )
                            for word in words:
                                positions.append(
                                    {
                                        "text": word["text"],
                                        "page": page_num,
                                        "x0": word["x0"],
                                        "y0": word["top"],
                                        "x1": word["x1"],
                                        "y1": word["bottom"],
                                    }
                                )

                        full_text = "\n\n".join(full_text_parts)

                        # Store positions for citation matching
                        doc_proc.set_pdf_blocks(conversation_id, positions)

                    # Check if this PDF is already in Snowflake parse results
                    snowflake_content = None
                    if client:
                        try:
                            existing_doc = (
                                doc_proc._check_existing_snowflake_document_sync(
                                    client, pdf_bytes, known_filename
                                )
                            )

                            if existing_doc and existing_doc.get("exists"):
                                # Get the parsed pages from the existing_doc response
                                pages = existing_doc.get("pages", [])

                                if pages:
                                    # Build content from Snowflake-parsed pages
                                    snowflake_content = "\n\n".join(
                                        f"[Page {p['page']}]\n{p['content']}"
                                        for p in pages
                                    )

                                    # Store Snowflake pages for additional citation matching
                                    doc_proc.set_document_pages(
                                        conversation_id,
                                        [
                                            {
                                                "page": p["page"],
                                                "content": p["content"],
                                                "file_name": existing_doc["file_name"],
                                            }
                                            for p in pages
                                        ],
                                    )

                                    # Update document source to indicate Snowflake content is available
                                    doc_source["has_snowflake_content"] = True
                                    doc_source["snowflake_file_name"] = existing_doc[
                                        "file_name"
                                    ]
                                    doc_source["snowflake_page_count"] = existing_doc[
                                        "page_count"
                                    ]
                                    doc_source["stage_path"] = existing_doc.get(
                                        "stage_path"
                                    )
                                    doc_proc.set_document_source(
                                        conversation_id, doc_source
                                    )
                        except Exception as e:
                            logger.debug(
                                "Snowflake check failed, using pdfplumber: %s", e
                            )

                    # Use Snowflake content if available, otherwise pdfplumber extraction
                    content_for_llm = (
                        snowflake_content if snowflake_content else full_text
                    )

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

    async def process_widget_response(
        self,
        widget,
        data: Any,
        conversation_id: str,
        client,
    ) -> dict[str, Any]:
        if self.is_document_widget(widget):
            known_filename = DocumentProcessor.extract_filename_from_widget(widget)
            parsed_data = self.parse_widget_data(
                data, conversation_id, client, known_filename
            )
            doc_proc = DocumentProcessor.instance()
            doc_proc.trigger_snowflake_upload_for_widget_pdf(client, conversation_id)
            return {"type": "document", "data": parsed_data}
        else:
            widget_uuid = str(widget.uuid) if hasattr(widget, "uuid") else "unknown"
            widget_name = (
                getattr(widget, "name", None)
                or getattr(widget, "title", None)
                or "widget"
            )
            widget_type = getattr(widget, "type", "generic")

            input_params = None
            if hasattr(widget, "params"):
                input_params = {p.name: p.current_value for p in widget.params}

            await self.store_widget_data(
                client,
                conversation_id,
                widget_uuid,
                widget_name,
                widget_type,
                input_params,
                data,
            )

            return {"type": "generic", "data": data, "widget_uuid": widget_uuid}

    async def handle_get_widget_data_tool_call(
        self,
        tool_args: dict[str, Any],
        widgets_primary: list | None,
        widgets_secondary: list | None,
        client,
        conversation_id: str,
    ) -> tuple[str, dict[str, Any] | None]:
        from openbb_ai import get_widget_data
        from openbb_ai.models import WidgetRequest

        widget_requests_arg = tool_args.get("widget_requests", [])
        if not widget_requests_arg:
            return "No widget requests provided.", None

        all_widgets = list(widgets_primary or []) + list(widgets_secondary or [])
        widget_requests = []

        for req_item in widget_requests_arg:
            widget_uuid = req_item.get("uuid") if isinstance(req_item, dict) else None
            if not widget_uuid:
                continue

            target_widget = None
            for w in all_widgets:
                if str(w.uuid) == widget_uuid:
                    target_widget = w
                    break

            if not target_widget:
                continue

            input_args = (
                {p.name: p.current_value for p in target_widget.params}
                if hasattr(target_widget, "params")
                else {}
            )

            if not self.is_document_widget(target_widget):
                cached_data = await self.get_cached_widget_data(
                    client,
                    conversation_id,
                    widget_uuid,
                    input_args,
                )

                if cached_data is not None:

                    return (
                        f"[Tool Result from get_widget_data]\nUsing cached data for widget {widget_uuid}",
                        {"cached": True, "data": cached_data},
                    )

            widget_request = WidgetRequest(
                widget=target_widget,
                input_arguments=input_args,
            )
            widget_requests.append(widget_request)

        if not widget_requests:
            return "All requested widgets have cached data.", None

        result = get_widget_data(widget_requests)

        result_str = str(
            result.model_dump() if hasattr(result, "model_dump") else result
        )
        result_dict = result.model_dump() if hasattr(result, "model_dump") else None

        return result_str, {"widget_data_request": result_dict} if result_dict else None
