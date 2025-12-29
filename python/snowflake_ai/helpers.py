"""Snowflake AI helpers."""

# flake8: noqa: PLR0911, PLR0912
# pylint: disable = R0911, R0912, R0914, R0915, R0917, C0103, C0415

import hashlib
import json
import re
import threading
from collections import defaultdict
from typing import Any, Iterable, Mapping

# Import DocumentProcessor singleton for document handling
from .document_processor import DocumentProcessor

# Expose document processor stores as module-level references
# These now reference the singleton's internal stores
_doc_proc = DocumentProcessor.instance()
pdf_text_blocks = _doc_proc.pdf_text_store
llm_referenced_quotes = _doc_proc.llm_quote_store
document_sources = _doc_proc.document_source_store
snowflake_document_pages = _doc_proc.snowflake_page_store

# Conversation deduplication caches (in-memory only)
_MESSAGE_SIGNATURE_CACHE: dict[str, set[str]] = defaultdict(set)
_SIGNATURE_LOCK = threading.Lock()

# Pattern to match citation markers:
# - Cortex native format: [cite:0], [cite:1], etc.
# - Our unique format: [|cite:0|], [|cite:1|], etc.
_CITE_MARKER_PATTERN = re.compile(r"\[(?:\|)?cite:\d+(?:\|)?\]")


def _normalize_content(value: str | None) -> str:
    """Normalize content for signature comparison.

    Strips citation markers [cite:N] and [|cite:N|] to prevent duplicates when the same
    response is stored both with and without citation markers.
    """
    if not value:
        return ""
    # Remove Cortex citation markers first
    value = _CITE_MARKER_PATTERN.sub("", value)
    # Collapse whitespace and strip
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


# ==============================================================================
# DOCUMENT PROCESSING FUNCTIONS - Now in DocumentProcessor singleton
# ==============================================================================
# All document processing functions have been moved to document_processor.py
# Use DocumentProcessor.instance() to access these methods:
#   - upload_pdf_bytes_to_snowflake
#   - remove_file_from_stage
#   - remove_file_from_stage_sync
#   - parse_widget_data
#   - handle_get_widget_data_tool_call
#   - store_widget_data_in_snowflake
#   - check_existing_snowflake_document
#   - load_snowflake_document_pages
#   - find_best_match_in_snowflake_pages
#   - extract_pdf_with_positions
#   - find_best_match
#   - parse_single_data_item
#   - extract_filename_from_stage_path
#   - extract_filename_from_widget
#   - find_quote_in_pdf_blocks
#   - extract_quotes_from_llm_response
# ==============================================================================


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
    # Validate input type to catch bugs early with clear error messages
    if not hasattr(sse_event, "event") or not hasattr(sse_event, "data"):
        raise TypeError(
            f"to_sse() expects an SSE event object with 'event' and 'data' attributes, "
            f"got {type(sse_event).__name__}: {sse_event!r}"
        )
    # Use model_dump_json() which returns a JSON string directly with proper UUID handling
    return {
        "event": sse_event.event,
        "data": sse_event.data.model_dump_json(),
    }


# ==============================================================================
# Document Processing Wrapper Functions (for backward compatibility)
# ==============================================================================
# These functions delegate to DocumentProcessor singleton methods


def find_quote_in_pdf_blocks(quote: str, conversation_id: str) -> dict | None:
    """Find a quote in the PDF text blocks and return its position data (delegates to DocumentProcessor)."""
    doc_proc = DocumentProcessor.instance()
    return doc_proc.find_quote_in_pdf_blocks(quote, conversation_id)


def extract_quotes_from_llm_response(response_text: str) -> list[tuple[str, int]]:
    """Extract citation references from LLM response (delegates to DocumentProcessor static method)."""
    return DocumentProcessor.extract_quotes_from_llm_response(response_text)


# ------------------------------------------------------------------
# Async and Iterator Utilities
# ------------------------------------------------------------------


async def run_in_thread(func, *args, **kwargs):
    """Wrap a blocking function to run in a background thread."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def is_iterator(value: object) -> bool:
    """Check if a value is an iterator."""
    from typing import Iterator

    return isinstance(value, Iterator)


def is_reasoning_event(event: object) -> bool:
    """Check if an event is a reasoning event."""
    if not isinstance(event, dict):
        return False

    if event.get("event") in {"reasoning", "reasoning_step"}:
        return True

    payload = event.get("data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return False

    if isinstance(payload, dict):
        payload_type = payload.get("type") or payload.get("event")
        return payload_type in {"reasoning", "reasoning_step"}

    return False


async def iterate_sync_generator(generator):
    """Consume a sync generator in a background thread and yield results."""
    import asyncio

    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def consume_generator():
        try:
            for chunk in generator:
                if chunk is not None:
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=consume_generator, daemon=True)
    thread.start()

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk


def extract_markdown_tables(text: str) -> tuple[str, list[dict]]:
    """Extract markdown tables from text and convert to structured data.

    Detects pipe-delimited markdown tables and extracts them as structured
    data suitable for AgGrid table artifacts.

    Parameters
    ----------
    text : str
        Text containing potential markdown tables

    Returns
    -------
    tuple[str, list[dict]]
        - Modified text with tables replaced by placeholders
        - List of extracted table dicts with keys:
          - 'data': list of row dicts (column_name: value)
          - 'name': auto-generated table name
          - 'placeholder': the placeholder string inserted in text
    """
    tables = []
    lines = text.split("\n")
    result_lines = []
    table_counter = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for potential table start (line with multiple pipes)
        if "|" in stripped and stripped.count("|") >= 2:
            table_lines = []
            table_start_idx = i

            # Collect consecutive table lines
            while i < len(lines):
                current = lines[i].strip()
                # Stop if empty line or no pipes
                if not current or "|" not in current:
                    break
                table_lines.append(current)
                i += 1

            # Need at least 2 lines (header + separator or header + data)
            if len(table_lines) >= 2:
                headers = []
                rows = []
                separator_idx = -1

                # Find separator line (contains only |, -, :, spaces)
                for idx, tline in enumerate(table_lines):
                    cleaned = (
                        tline.replace("|", "")
                        .replace("-", "")
                        .replace(":", "")
                        .replace(" ", "")
                    )
                    if not cleaned or all(c in "-:|" for c in tline.replace(" ", "")):
                        separator_idx = idx
                        break

                def parse_table_row(line: str) -> list[str]:
                    """Parse a markdown table row, handling leading/trailing pipes."""
                    # Strip leading/trailing whitespace and pipes
                    line = line.strip()
                    if line.startswith("|"):
                        line = line[1:]
                    if line.endswith("|"):
                        line = line[:-1]
                    # Split and clean each cell
                    return [cell.strip() for cell in line.split("|")]

                # Parse headers
                header_idx = separator_idx - 1 if separator_idx > 0 else 0
                if 0 <= header_idx < len(table_lines):
                    headers = parse_table_row(table_lines[header_idx])
                    # Filter out empty headers
                    headers = [h for h in headers if h]

                # Parse data rows
                data_start = separator_idx + 1 if separator_idx >= 0 else 1
                for row_line in table_lines[data_start:]:
                    cleaned = (
                        row_line.replace("|", "")
                        .replace("-", "")
                        .replace(":", "")
                        .replace(" ", "")
                    )
                    if not cleaned:
                        continue
                    row_values = parse_table_row(row_line)
                    # Pad or trim to match headers
                    if headers:
                        while len(row_values) < len(headers):
                            row_values.append("")
                        row_values = row_values[: len(headers)]
                    if row_values:
                        rows.append(row_values)

                # Only create artifact if we have headers and rows
                if headers and rows and len(rows) >= 1:
                    # Convert to list of dicts
                    data = []
                    for row in rows:
                        row_dict = {}
                        for col_idx, header in enumerate(headers):
                            value = row[col_idx] if col_idx < len(row) else ""
                            # Try to convert numeric values
                            try:
                                # Remove commas from numbers
                                clean_val = (
                                    value.replace(",", "")
                                    .replace("$", "")
                                    .replace("%", "")
                                )
                                if "." in clean_val:
                                    row_dict[header] = float(clean_val)
                                else:
                                    row_dict[header] = int(clean_val)
                            except (ValueError, AttributeError):
                                row_dict[header] = value
                        data.append(row_dict)

                    table_counter += 1
                    placeholder = f"[TABLE_ARTIFACT_{table_counter}]"
                    table_name = f"Extracted Table {table_counter}"

                    tables.append(
                        {
                            "data": data,
                            "name": table_name,
                            "placeholder": placeholder,
                            "headers": headers,
                        }
                    )

                    # Replace table with placeholder
                    result_lines.append(placeholder)
                    continue
                else:
                    # Not a valid table, keep original lines
                    result_lines.extend(lines[table_start_idx:i])
                    continue
            else:
                # Not enough lines for a table
                result_lines.append(line)
                i += 1
                continue
        else:
            result_lines.append(line)
            i += 1

    return "\n".join(result_lines), tables


def format_tool_overview(tool_defs: list[dict] | None) -> str:
    """Create a human-readable summary of available tools."""

    if not tool_defs:
        return ""

    lines: list[str] = []
    for tool in tool_defs:
        if not isinstance(tool, dict):
            continue

        function = tool.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if not name:
            continue

        description = (function.get("description") or "").strip()
        parameters = function.get("parameters")
        arg_bits: list[str] = []

        if isinstance(parameters, dict):
            props = parameters.get("properties")
            if isinstance(props, dict):
                for param_name, schema in props.items():
                    if not isinstance(schema, dict):
                        continue
                    param_text = param_name
                    param_type = schema.get("type")
                    param_desc = (schema.get("description") or "").strip()
                    if param_type:
                        param_text += f" ({param_type})"
                    if param_desc:
                        param_text += f": {param_desc}"
                    arg_bits.append(param_text)

        arg_text = f" Args: {'; '.join(arg_bits)}" if arg_bits else ""
        lines.append(f"- {name}: {description}{arg_text}".strip())

    return "\n".join(lines)
