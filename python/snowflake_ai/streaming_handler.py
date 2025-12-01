"""LLM streaming response handler for Snowflake AI."""

import asyncio
import json
import os
import re
import threading
from collections.abc import Iterator

from openbb_ai import citations, message_chunk
from openbb_ai.models import Citation, CitationHighlightBoundingBox, SourceInfo

from ._snowflake_ai import FunctionCall, SnowflakeAI, ToolCall
from .document_processor import DocumentProcessor
from .helpers import to_sse
from .logger import get_logger

logger = get_logger(__name__)


def cleanup_text(text: str) -> str:
    """Remove extra backslashes from text."""
    return re.sub(r"\\([_`])", r"\1", text)


def sanitize_citation_text(text: str) -> str:
    """Sanitize text for citation matching by removing markdown formatting.

    This ensures that text with markdown blockquotes, emphasis, etc. can still
    match against the original PDF text content.
    """
    # Remove citation markers [N] that pollute the text from previous citations
    text = re.sub(r"\s*\[\d+\]\s*", " ", text)

    # Remove markdown table pipes and clean up table formatting
    text = re.sub(r"\s*\|\s*", " ", text)  # Replace | with space

    # Remove markdown blockquote markers
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

    # Remove markdown list markers (bullets, numbers) - more aggressive
    text = re.sub(r"^[-*+•]\s+", "", text, flags=re.MULTILINE)  # Various bullet types
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)  # 1. numbered lists
    text = re.sub(r"\s+[-*+•]\s+", " ", text)  # Mid-line bullets

    # Remove markdown bold/italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
    text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
    text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
    text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

    # Remove markdown code backticks
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove quotes that might be added
    text = re.sub(r'^["\']+|["\']+$', "", text)

    # Remove extra whitespace and normalize (also removes duplicate spaces)
    text = " ".join(text.split())

    return text.strip()


def extract_quoted_text_for_citation(text_before_citation: str) -> str | None:
    """Extract the quoted text immediately before a citation marker.

    Citations should follow verbatim quoted text from the document.
    This function extracts text in "double quotes" or 'single quotes'
    that appears near the end of the text before [N].

    Returns the quoted text without quotes, or None if no quote found.
    """
    if not text_before_citation:
        return None

    # Look for quoted text near the end (last 500 chars to be safe)
    search_text = (
        text_before_citation[-500:]
        if len(text_before_citation) > 500
        else text_before_citation
    )

    # Find ALL quoted strings in the search text, take the LAST one (closest to citation)
    # Match "quoted text" or 'quoted text' - at least 3 words
    double_quotes = re.findall(r'"([^"]{5,})"', search_text)
    single_quotes = re.findall(r"'([^']{5,})'", search_text)

    # Prefer double quotes, then single quotes
    all_quotes = double_quotes + single_quotes

    if not all_quotes:
        return None

    # Return the LAST quoted text found (closest to the citation marker)
    last_quote = all_quotes[-1].strip()

    # Validate it looks like real document text (not just punctuation or numbers)
    if len(last_quote) < 5:
        return None

    # Must have at least one letter
    if not re.search(r"[a-zA-Z]", last_quote):
        return None

    return last_quote


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def iterate_sync_generator(generator: Iterator):
    """Consume a sync generator in a background thread and yield results."""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()
    thread_exception = None

    def consume_generator():
        nonlocal thread_exception
        try:
            for chunk in generator:
                if chunk is not None:
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
        except Exception as e:
            thread_exception = e
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Error in generator thread: %s", e)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=consume_generator, daemon=True)
    thread.start()

    # Add timeout to prevent infinite waiting
    while True:
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            if chunk is None:
                if thread_exception is not None:
                    raise thread_exception
                break
            yield chunk
        except asyncio.TimeoutError:
            logger.debug("Stream timeout - terminating")
            break


class CitationMatchError(RuntimeError):
    """Raised when an inline citation cannot be matched to document text."""


async def stream_llm_with_tools(
    client: SnowflakeAI,
    messages: list[tuple[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    tools: list | None = None,
    conv_id: str = "default",
    widget=None,
    widget_input_args=None,
):
    """
    Stream LLM response and yield text chunks with inline citations.

    Yields chunks of ('text', str), ('citation', Citation), or ('complete', {'text': str, 'tool_calls': list, 'usage': dict})
    """
    doc_proc = DocumentProcessor.instance()

    def estimate_tokens(text: str) -> int:
        """Estimate token count - roughly 4 characters per token for English."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    # Filter out empty messages and validate content - Cortex API requires valid content
    filtered_messages = []
    prompt_text = ""  # Track prompt text for token estimation
    for role, content in messages:
        if not content or not content.strip():
            continue
        # Ensure role is valid
        if role not in ("system", "user", "assistant"):
            if role in ("human",):
                role = "user"
            elif role in ("ai",):
                role = "assistant"
            else:
                continue  # Skip invalid roles
        filtered_messages.append((role, content))
        prompt_text += content + " "  # Accumulate for token estimation

    # DEBUG: Print message sequence to identify invalid sequences
    if os.environ.get("SNOWFLAKE_DEBUG"):
        logger.debug("=== Message sequence being sent to Cortex ===")
        for i, (role, content) in enumerate(filtered_messages):
            preview = (
                content[:100].replace("\n", " ") + "..."
                if len(content) > 100
                else content.replace("\n", " ")
            )
            logger.debug(f"  [{i}] {role}: {preview}")
        logger.debug("=== End message sequence ===")

    # Fix message sequence - Cortex requires alternating user/assistant (after system)
    # Merge consecutive same-role messages to prevent "invalid message role sequence" error
    fixed_messages = []
    for role, content in filtered_messages:
        if not fixed_messages:
            fixed_messages.append((role, content))
        elif role == "system":
            # Merge system messages into the first one
            if fixed_messages[0][0] == "system":
                fixed_messages[0] = ("system", fixed_messages[0][1] + "\n\n" + content)
            else:
                fixed_messages.insert(0, (role, content))
        elif fixed_messages[-1][0] == role:
            # Consecutive same role - merge
            prev_content = fixed_messages[-1][1]
            fixed_messages[-1] = (role, prev_content + "\n\n" + content)
        else:
            fixed_messages.append((role, content))

    # CRITICAL FIX: Cortex requires system -> user -> assistant pattern
    # If first non-system message is assistant, we need to insert a user message
    # Also ensure the sequence properly alternates
    corrected_messages = []
    for i, (role, content) in enumerate(fixed_messages):
        if role == "system":
            corrected_messages.append((role, content))
        elif role == "assistant":
            # Check if previous non-system message was user
            prev_non_system = None
            for prev_role, _ in reversed(corrected_messages):
                if prev_role != "system":
                    prev_non_system = prev_role
                    break

            if prev_non_system is None:
                # First non-system message is assistant - need a user message first
                # Insert a placeholder user message (the assistant response was to a slash command)
                corrected_messages.append(("user", "[System command executed]"))
            elif prev_non_system == "assistant":
                # Consecutive assistant - merge with previous
                for j in range(len(corrected_messages) - 1, -1, -1):
                    if corrected_messages[j][0] == "assistant":
                        corrected_messages[j] = (
                            "assistant",
                            corrected_messages[j][1] + "\n\n" + content,
                        )
                        break
                continue

            corrected_messages.append((role, content))
        elif role == "user":
            # Check if previous non-system message was assistant or none
            prev_non_system = None
            for prev_role, _ in reversed(corrected_messages):
                if prev_role != "system":
                    prev_non_system = prev_role
                    break

            if prev_non_system == "user":
                # Consecutive user - merge with previous
                for j in range(len(corrected_messages) - 1, -1, -1):
                    if corrected_messages[j][0] == "user":
                        corrected_messages[j] = (
                            "user",
                            corrected_messages[j][1] + "\n\n" + content,
                        )
                        break
                continue

            corrected_messages.append((role, content))
        else:
            corrected_messages.append((role, content))

    filtered_messages = corrected_messages

    if not filtered_messages:
        yield ("text", "No valid messages to process.")
        yield (
            "complete",
            {"text": "No valid messages to process.", "tool_calls": [], "usage": {}},
        )
        return

    # Estimate prompt tokens upfront
    estimated_prompt_tokens = estimate_tokens(prompt_text)

    # Don't pass tools to the API - they're described in the system prompt
    # The LLM will output tool calls and we detect them from the event type
    def _should_retry_cortex_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        transient_markers = (
            "internal error",
            "timed out",
            "temporarily unavailable",
            "service unavailable",
            "please retry",
            "429",
            "500",
            "502",
            "503",
        )
        return any(marker in message for marker in transient_markers)

    last_exception = None
    response_stream = None
    for attempt in range(3):
        try:
            response_stream = await run_in_thread(
                client.chat_stream,
                messages=filtered_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
            )
            break
        except RuntimeError as e:  # pragma: no cover - network path
            last_exception = e
            if "tool calling is not supported" in str(e):
                # Retry immediately without counting against transient budget
                response_stream = await run_in_thread(
                    client.chat_stream,
                    messages=filtered_messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=None,
                )
                last_exception = None
                break

            if not _should_retry_cortex_error(e) or attempt == 2:
                raise

            # Transient Cortex hiccup—back off briefly before retrying
            backoff_delay = 1 + attempt  # 1s, 2s, ...
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.debug(
                    "Cortex error '%s'. Retry %d/3 after %ds",
                    e,
                    attempt + 1,
                    backoff_delay,
                )
            await asyncio.sleep(backoff_delay)

    if response_stream is None:
        # Should never happen, but surface last exception for clarity
        if last_exception:
            raise last_exception
        raise RuntimeError("Failed to initialize Cortex response stream")

    full_text = ""
    tool_calls_in_progress = {}
    buffer = ""
    citations_created = {}  # Track which citation numbers we've created
    citation_objects = {}  # Store citation objects for inline replacement of [N]
    final_usage = None  # Track usage stats
    MAX_CITATIONS = 10  # Allow up to 10 citations per response

    # State for handling <think>...</think> blocks from deepseek-r1
    inside_think_block = False
    think_buffer = ""

    def _get_attr(source, *names, default=None):
        for name in names:
            if isinstance(source, dict) and name in source:
                value = source[name]
                if value is not None:
                    return value
            value = getattr(source, name, None)
            if value is not None:
                return value
        return default

    def _ensure_arg_string(arg_piece) -> str:
        if arg_piece is None:
            return ""
        if isinstance(arg_piece, str):
            return arg_piece
        try:
            return json.dumps(arg_piece)
        except TypeError:
            return str(arg_piece)

    def upsert_tool_call_entry(call_id: str | None, default_name: str | None = None):
        if not call_id:
            call_id = f"tool_call_{len(tool_calls_in_progress) + 1}"
        entry = tool_calls_in_progress.setdefault(
            call_id,
            {"id": call_id, "name": "", "arguments": ""},
        )
        if default_name and not entry["name"]:
            entry["name"] = default_name
        return entry

    def handle_tool_delta(part) -> bool:
        if part is None:
            return False

        part_type = _get_attr(part, "type", "delta_type", "content_type")
        function_obj = _get_attr(part, "function")

        if part_type in {"tool_use", "tool_call", "tool"} or function_obj:
            call_id = _get_attr(
                part,
                "id",
                "tool_use_id",
                "toolUseId",
                "call_id",
                "callId",
            )
            default_name = _get_attr(part, "name")
            arguments_piece = _get_attr(part, "arguments", "input")

            if function_obj:
                default_name = default_name or _get_attr(function_obj, "name")
                fn_args = _get_attr(function_obj, "arguments")
                if fn_args is not None:
                    arguments_piece = (arguments_piece or "") + _ensure_arg_string(
                        fn_args
                    )

            if arguments_piece:
                arguments_piece = _ensure_arg_string(arguments_piece)

            entry = upsert_tool_call_entry(call_id, default_name)
            if arguments_piece:
                entry["arguments"] += arguments_piece
            return True

        if part_type in {"function_call", "function"} or (
            _get_attr(part, "name") and _get_attr(part, "arguments") is not None
        ):
            call_id = _get_attr(part, "id") or "function_call"
            entry = upsert_tool_call_entry(call_id, _get_attr(part, "name"))
            arguments_piece = _get_attr(part, "arguments")
            if arguments_piece:
                entry["arguments"] += _ensure_arg_string(arguments_piece)
            return True

        return False

    try:
        async for chunk in iterate_sync_generator(response_stream):
            if os.environ.get("SNOWFLAKE_DEBUG"):
                # Try different ways to extract data from the chunk
                if hasattr(chunk, "__dict__"):
                    logger.debug("RAW SSE: %s", chunk.__dict__)
                elif hasattr(chunk, "to_dict"):
                    logger.debug("RAW SSE: %s", chunk.to_dict())
                else:
                    # Try to serialize it as JSON if possible
                    dumped = (
                        chunk.to_json_string()
                        if hasattr(chunk, "to_json_string")
                        else str(chunk.dict())
                    )
                    logger.debug("RAW SSE: %s", dumped)

            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage

                # Extract usage fields - handle both dict and object formats
                if isinstance(usage_data, dict) and usage_data.get("total_tokens"):
                    final_usage = {
                        "prompt_tokens": usage_data.get("prompt_tokens", 0),
                        "completion_tokens": usage_data.get("completion_tokens", 0),
                        "total_tokens": usage_data.get("total_tokens", 0),
                        "cache_read_input_tokens": usage_data.get(
                            "cache_read_input_tokens", 0
                        ),
                    }
                else:
                    final_usage = {
                        "prompt_tokens": getattr(usage_data, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            usage_data, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(usage_data, "total_tokens", 0),
                        "cache_read_input_tokens": getattr(
                            usage_data, "cache_read_input_tokens", 0
                        ),
                    }

            if not (hasattr(chunk, "choices") and chunk.choices):
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            delta_type = getattr(delta, "delta_type", None) or getattr(
                delta, "type", None
            )

            tool_calls_payload = _get_attr(delta, "tool_calls", "toolCalls")
            if tool_calls_payload:
                payload_list = (
                    tool_calls_payload
                    if isinstance(tool_calls_payload, (list, tuple))
                    else [tool_calls_payload]
                )
                for call_part in payload_list:
                    if not handle_tool_delta(call_part):
                        nested_function = _get_attr(call_part, "function")
                        if nested_function:
                            handle_tool_delta(nested_function)

            function_call_delta = _get_attr(delta, "function_call", "functionCall")
            if function_call_delta:
                handle_tool_delta(function_call_delta)

            content_list = _get_attr(delta, "content", "content_list")
            if isinstance(content_list, list):
                for content_part in content_list:
                    if handle_tool_delta(content_part):
                        continue

            if delta_type in {"tool_use", "tool_call", "tool"}:
                handle_tool_delta(delta)
                continue

            if delta_type == "text" or delta_type == "message":
                text_piece = getattr(delta, "text", None) or getattr(
                    delta, "content", None
                )

                if not text_piece and getattr(delta, "content_list", None):
                    parts = []
                    for item in delta.content_list:
                        t = None
                        if isinstance(item, dict):
                            t = item.get("text")
                        else:
                            t = getattr(item, "text", None)
                        if t:
                            parts.append(t)
                    text_piece = "".join(parts)

                if text_piece:
                    # Handle <think>...</think> blocks from deepseek-r1
                    # Accumulate think content silently, then emit as ONE reasoning event when block closes
                    # This prevents token-by-token reasoning events

                    remaining_text = text_piece
                    output_text = ""

                    while remaining_text:
                        if inside_think_block:
                            # Look for closing </think> tag
                            end_idx = remaining_text.find("</think>")
                            if end_idx != -1:
                                # Found closing tag - accumulate final think content
                                final_think_content = remaining_text[:end_idx]
                                if final_think_content:
                                    think_buffer += final_think_content
                                # Emit the COMPLETE think block as ONE reasoning event
                                if think_buffer.strip():
                                    yield ("reasoning_complete", think_buffer.strip())
                                think_buffer = ""
                                inside_think_block = False
                                remaining_text = remaining_text[
                                    end_idx + 8 :
                                ]  # Skip </think>
                            else:
                                # No closing tag yet - just accumulate, don't yield
                                think_buffer += remaining_text
                                remaining_text = ""
                        else:
                            # Look for opening <think> tag
                            start_idx = remaining_text.find("<think>")
                            if start_idx != -1:
                                # Found opening tag - output text before it
                                output_text += remaining_text[:start_idx]
                                inside_think_block = True
                                think_buffer = ""
                                remaining_text = remaining_text[
                                    start_idx + 7 :
                                ]  # Skip <think>
                            else:
                                # Check for partial <think tag at end (streaming boundary)
                                # Buffer last 7 chars to handle split tags
                                partial_tag_check = False
                                for i in range(1, min(7, len(remaining_text) + 1)):
                                    if remaining_text.endswith("<think>"[:i]):
                                        # Potential partial tag - hold back these chars
                                        output_text += remaining_text[:-i]
                                        think_buffer = remaining_text[-i:]
                                        partial_tag_check = True
                                        break

                                if not partial_tag_check:
                                    # No think tag - output everything
                                    output_text += remaining_text
                                remaining_text = ""

                    # Only process non-think text through citation handling
                    if output_text:
                        cleaned = cleanup_text(output_text)
                        buffer += cleaned
                        full_text += cleaned

                    # Process buffer for citations
                    while True:
                        # Look for [N] pattern
                        match = re.search(r"\[(\d+)\]", buffer)
                        if not match:
                            # No citation found yet - check if we should yield accumulated buffer
                            # Only yield if buffer is getting long (>200 chars) to allow [N] patterns to accumulate
                            # Keep last 10 chars to prevent pattern splitting at boundary
                            if len(buffer) > 200:
                                yield ("text", buffer[:-10])
                                buffer = buffer[-10:]
                            # Otherwise, keep accumulating buffer across streaming chunks
                            break

                        citation_num = int(match.group(1))
                        text_before = buffer[: match.start()]

                        if text_before:
                            yield ("text", text_before)

                        # ENFORCE CITATION CAP FIRST - before any processing
                        # This counts SUCCESSFUL citations only (ones in citation_objects)
                        if len(citation_objects) >= MAX_CITATIONS:
                            # Already have max citations - silently drop all remaining markers
                            buffer = buffer[match.end() :]
                            continue

                        # Skip duplicate citation numbers
                        if citation_num in citations_created:
                            buffer = buffer[match.end() :]
                            continue

                        if widget:
                            citations_created[citation_num] = True

                            # Check for available position data sources
                            has_pdf_positions = bool(doc_proc.get_pdf_blocks(conv_id))
                            has_snowflake = bool(doc_proc.get_document_pages(conv_id))

                            if has_pdf_positions or has_snowflake:
                                # Get what the LLM wrote before this citation for context matching
                                recent_text = (full_text + text_before).strip()

                                # Sanitize markdown to help match against PDF text
                                sanitized_text = sanitize_citation_text(recent_text)

                                # Take last ~300 chars before citation for matching
                                if len(sanitized_text) > 300:
                                    current_sentence = sanitized_text[-300:]
                                else:
                                    current_sentence = sanitized_text

                                # Clean up if we cut mid-word
                                if len(sanitized_text) > 300:
                                    space_idx = current_sentence.find(" ")
                                    if space_idx > 0 and space_idx < 50:
                                        current_sentence = current_sentence[
                                            space_idx + 1 :
                                        ]

                                logger.debug(
                                    "Citation [%d] - Context for matching (%d chars): '%s...'",
                                    citation_num,
                                    len(current_sentence),
                                    current_sentence[:80],
                                )

                                # Page explicit hint matching
                                page_hint = None
                                page_match = re.search(
                                    r"page\s+(\d+)", current_sentence, re.IGNORECASE
                                )
                                if page_match:
                                    page_hint = int(page_match.group(1))

                                selected_position = None

                                # Try PDF positions first (most precise coordinates)
                                if has_pdf_positions:
                                    pdf_positions = doc_proc.get_pdf_blocks(conv_id)
                                    if pdf_positions:
                                        selected_position = doc_proc.find_best_match(
                                            current_sentence,
                                            pdf_positions,
                                            preferred_page=page_hint,
                                            require_exact_phrases=True,  # STRICT matching for quotes
                                        )
                                        if selected_position:
                                            selected_position["source"] = (
                                                "pdf_positions"
                                            )

                                # Fall back to Snowflake document pages
                                if not selected_position and has_snowflake:
                                    selected_position = (
                                        doc_proc.find_best_match_in_snowflake_pages(
                                            current_sentence,
                                            conv_id,
                                            preferred_page=page_hint,
                                        )
                                    )

                                # CRITICAL: If no match found, DROP the citation
                                # We require actual document matches for citations
                                if not selected_position or not selected_position.get(
                                    "text"
                                ):
                                    logger.warning(
                                        "Citation [%d] DROPPED - context text not found in document: '%s...'",
                                        citation_num,
                                        (
                                            current_sentence[:50]
                                            if current_sentence
                                            else "N/A"
                                        ),
                                    )
                                    buffer = buffer[match.end() :]
                                    continue

                                if conv_id:
                                    doc_proc.record_llm_quote(
                                        conv_id, current_sentence, citation_num
                                    )

                                # Create citation details with capitalized keys
                                doc_name = (
                                    (widget_input_args or {}).get("file_name")
                                    or widget.name
                                    or "document"
                                )

                                # Use page number as the citation label instead of generic name
                                citation_label = f"(pg. {selected_position['page']})"

                                citation_details = {
                                    "Page": selected_position["page"],
                                    "Reference": selected_position["text"],
                                    "Filename": doc_name,
                                }

                                # Check if this is a PDF document (only PDFs support bounding boxes)
                                file_name = (widget_input_args or {}).get(
                                    "file_name", ""
                                )
                                is_pdf = str(file_name).lower().endswith(".pdf")

                                # Build bounding boxes only when we have REAL coordinates from matching
                                # If find_best_match returned page-only (no coords), skip highlighting
                                all_bboxes = None
                                if is_pdf:
                                    # Check if position has actual coordinates (not just page)
                                    has_coords = all(
                                        k in selected_position
                                        and selected_position[k] is not None
                                        for k in ["x0", "top", "x1", "bottom"]
                                    )
                                    has_text = selected_position.get("text", "").strip()

                                    if has_coords and has_text:
                                        try:
                                            x0 = float(selected_position["x0"])
                                            top = float(selected_position["top"])
                                            x1 = float(selected_position["x1"])
                                            bottom = float(selected_position["bottom"])

                                            # Validate coordinates are reasonable
                                            # Skip if coords look like default/fabricated values
                                            coords_valid = (
                                                x0 >= 0
                                                and top >= 0
                                                and x1 > x0
                                                and bottom > top
                                                # Not fabricated defaults
                                                and not (
                                                    x0 == 50
                                                    and top == 50
                                                    and x1 == 550
                                                    and bottom == 100
                                                )
                                                and not (
                                                    x0 == 50
                                                    and top == 100
                                                    and x1 == 550
                                                    and bottom == 200
                                                )
                                                and not (
                                                    x0 == 50
                                                    and top == 100
                                                    and x1 == 550
                                                    and bottom == 150
                                                )
                                            )

                                            if coords_valid:
                                                primary_bbox = (
                                                    CitationHighlightBoundingBox(
                                                        text=selected_position["text"],
                                                        page=selected_position["page"],
                                                        x0=x0,
                                                        top=top,
                                                        x1=x1,
                                                        bottom=bottom,
                                                    )
                                                )
                                                all_bboxes = [[primary_bbox]]
                                                logger.debug(
                                                    "Citation [%d] has valid highlight on page %d",
                                                    citation_num,
                                                    selected_position["page"],
                                                )
                                            else:
                                                logger.debug(
                                                    "Citation [%d] page %d - coords invalid, page-only citation",
                                                    citation_num,
                                                    selected_position.get("page"),
                                                )
                                        except (ValueError, TypeError) as e:
                                            logger.debug(
                                                "Citation [%d] - coordinate parse error, page-only: %s",
                                                citation_num,
                                                e,
                                            )
                                            all_bboxes = None
                                    else:
                                        # No coords or no text - this is a page-only citation
                                        logger.debug(
                                            "Citation [%d] page %d - no match text, page-only citation",
                                            citation_num,
                                            selected_position.get("page"),
                                        )

                                citation_obj = Citation(
                                    source_info=SourceInfo(
                                        type="widget",
                                        uuid=widget.uuid,
                                        origin=widget.origin,
                                        widget_id=str(widget.widget_id),
                                        name=citation_label,
                                        citable=True,
                                        metadata={
                                            "input_args": (
                                                {
                                                    p.name: p.current_value
                                                    for p in widget.params
                                                }
                                                if hasattr(widget, "params")
                                                else {}
                                            )
                                        },
                                    ),
                                    details=[citation_details],
                                    quote_bounding_boxes=all_bboxes,
                                )

                                # Store citation object for this number
                                citation_objects[citation_num] = citation_obj
                            else:
                                # No PDF positions or Snowflake pages available - skip citation
                                logger.warning(
                                    "Citation [%d] SKIPPED - no document data available for matching",
                                    citation_num,
                                )

                        # ALWAYS yield citation inline at this position (replacing [N])
                        if citation_num in citation_objects:
                            yield (
                                "citation",
                                citation_objects[citation_num],
                            )
                        elif os.environ.get("SNOWFLAKE_DEBUG"):
                            logger.debug(
                                "Citation [%d] not found in citation_objects",
                                citation_num,
                            )

                        # Continue with rest of buffer
                        buffer = buffer[match.end() :]

    except CitationMatchError as citation_error:
        logger.error("Citation match failure: %s", citation_error)
        yield ("fatal", str(citation_error))
        return
    except Exception as e:
        logger.error("Stream error: %s", e, exc_info=True)
        # Still yield what we have
        if buffer:
            yield ("text", buffer)
            buffer = ""

    # Yield any remaining buffer - no citation processing here since it should have been done in the loop
    if buffer:
        yield ("text", buffer)

    # Build tool calls
    tool_calls = []
    if tool_calls_in_progress:
        for _, info in tool_calls_in_progress.items():
            if (
                info["name"] and info["arguments"]
            ):  # Only add if we have both name and args
                tool_calls.append(
                    ToolCall(
                        id=info["id"],
                        tool_type="function",
                        function=FunctionCall(
                            name=info["name"],
                            arguments=info["arguments"],
                        ),
                    )
                )

    # If no usage from API, estimate tokens
    if not final_usage or not final_usage.get("total_tokens"):
        estimated_completion_tokens = estimate_tokens(full_text)
        final_usage = {
            "prompt_tokens": estimated_prompt_tokens,
            "completion_tokens": estimated_completion_tokens,
            "total_tokens": estimated_prompt_tokens + estimated_completion_tokens,
            "estimated": True,  # Flag that these are estimates
        }

    yield (
        "complete",
        {"text": full_text, "tool_calls": tool_calls, "usage": final_usage},
    )


async def generate_sse_events(stream_generator, stream_state: dict):
    """Consume the stream generator and yield SSE events."""
    from openbb_ai import reasoning_step

    stream_state.setdefault("citation_count", 0)
    stream_state.setdefault("citation_summaries", [])
    stream_state.setdefault("fatal_error", None)

    try:
        async for event_type, data in stream_generator:
            if event_type == "text" and isinstance(data, str):
                yield to_sse(message_chunk(data))
                stream_state["full_text"] += data
            elif event_type == "reasoning_complete" and isinstance(data, str):
                # Emit complete think block as a single reasoning event
                yield to_sse(reasoning_step(data, event_type="INFO"))
            elif event_type == "citation":
                # Citation model uses 'details' list, not 'extra_details'
                if hasattr(data, "details") and data.details:
                    yield to_sse(citations([data]))
                stream_state["citation_count"] = (
                    stream_state.get("citation_count", 0) + 1
                )
                # Extract first dict from details list
                summary_payload = data.details[0] if data.details else {}
                summary_payload.setdefault(
                    "citation_id", getattr(data, "citation_id", None)
                )
                stream_state.setdefault("citation_summaries", []).append(
                    summary_payload
                )
            elif event_type == "fatal":
                failure_message = str(data)
                stream_state["fatal_error"] = failure_message
                stream_state["full_text"] = ""
                yield to_sse(message_chunk(f"❌ Citation failure: {failure_message}"))
                break
            elif event_type == "complete":
                if isinstance(data, dict):
                    stream_state["full_text"] = data.get("text", "")
                    stream_state["tool_calls"] = data.get("tool_calls", [])
                    stream_state["usage"] = data.get("usage")  # Capture usage stats
                else:
                    stream_state["full_text"] = str(data)
                    stream_state["tool_calls"] = []
    except Exception as e:
        logger.error("SSE generator error: %s", e, exc_info=True)
        # Mark stream as failed so callers stop waiting
        stream_state["fatal_error"] = str(e)
        stream_state["full_text"] = stream_state.get("full_text", "")
        # Yield error message
        yield to_sse(message_chunk(f"Stream error: {str(e)}"))
    finally:
        # Ensure stream_state always has termination markers
        if "fatal_error" not in stream_state:
            stream_state.setdefault("full_text", "")
        stream_state["stream_completed"] = True
