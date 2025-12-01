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

# Pattern to match Cortex citation markers like [cite:0], [cite:1], etc.
_CITE_MARKER_PATTERN = re.compile(r"\[cite:\d+\]")


def _normalize_content(value: str | None) -> str:
    """Normalize content for signature comparison.

    Strips Cortex citation markers [cite:N] to prevent duplicates when the same
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
