"""LLM streaming response handler for Snowflake AI."""

import asyncio
import json
import os
import re
import threading
from collections.abc import Iterator

from ._snowflake_ai import FunctionCall, SnowflakeAI, ToolCall


def cleanup_text(text: str) -> str:
    """Remove extra backslashes from text."""
    return re.sub(r"\\([_`])", r"\1", text)


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
                print(f"[DEBUG] Error in generator thread: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    thread = threading.Thread(target=consume_generator, daemon=True)
    thread.start()

    # Add timeout to prevent infinite waiting
    while True:
        try:
            chunk = await asyncio.wait_for(queue.get(), timeout=30.0)
            if chunk is None:
                if thread_exception:
                    raise thread_exception
                break
            yield chunk
        except asyncio.TimeoutError:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print("[DEBUG] Stream timeout - terminating")
            break


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
    from openbb_ai import cite
    from openbb_ai.models import CitationHighlightBoundingBox
    from .helpers import (
        pdf_text_blocks,
        find_best_match,
        llm_referenced_quotes,
        snowflake_document_pages,
        document_sources,
        find_best_match_in_snowflake_pages,
    )

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

    if not filtered_messages:
        yield ("text", "No valid messages to process.")
        yield (
            "complete",
            {"text": "No valid messages to process.", "tool_calls": [], "usage": {}},
        )
        return

    # Estimate prompt tokens upfront
    estimated_prompt_tokens = estimate_tokens(prompt_text)

    # Debug: log message count
    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Sending {len(filtered_messages)} messages to Cortex API")

    # Don't pass tools to the API - they're described in the system prompt
    # The LLM will output tool calls and we detect them from the event type
    try:
        response_stream = await run_in_thread(
            client.chat_stream,
            messages=filtered_messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=None,
        )
    except RuntimeError as e:
        if "tool calling is not supported" in str(e):
            response_stream = await run_in_thread(
                client.chat_stream,
                messages=filtered_messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
            )
        else:
            raise

    full_text = ""
    tool_calls_in_progress = {}
    buffer = ""
    citations_created = {}  # Track which citation numbers we've created
    final_usage = None  # Track usage stats

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
                    print(f"[RAW SSE DATA]: {chunk.__dict__}")
                elif hasattr(chunk, "to_dict"):
                    print(f"[RAW SSE DATA]: {chunk.to_dict()}")
                else:
                    # Try to serialize it as JSON if possible
                    dumped = (
                        chunk.to_json_string()
                        if hasattr(chunk, "to_json_string")
                        else str(chunk.dict())
                    )
                    print(f"[RAW SSE DATA]: {dumped}")

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
                    cleaned = cleanup_text(text_piece)
                    buffer += cleaned
                    full_text += cleaned

                    # Process buffer for citations
                    while True:
                        # Look for [N] pattern
                        match = re.search(r"\[(\d+)\]", buffer)
                        if not match:
                            # No citation found, yield accumulated buffer
                            if len(buffer) > 100:  # Yield if buffer gets long
                                yield ("text", buffer)
                                buffer = ""
                            break

                        # Found a citation!
                        citation_num = int(match.group(1))

                        # Yield text BEFORE the citation
                        text_before = buffer[: match.start()]
                        if text_before:
                            yield ("text", text_before)

                        # Yield the [N] itself
                        yield ("text", f"[{citation_num}]")

                        # Create and yield citation if we have widget info
                        if widget and citation_num not in citations_created:
                            citations_created[citation_num] = True

                            # Check for available position data sources
                            has_pdfplumber = (
                                conv_id in pdf_text_blocks and pdf_text_blocks[conv_id]
                            )
                            has_snowflake = (
                                conv_id in snowflake_document_pages
                                and snowflake_document_pages[conv_id]
                            )

                            if has_pdfplumber or has_snowflake:
                                # Get what the LLM wrote before this citation
                                recent_text = (full_text + text_before).strip()

                                # Get the last sentence
                                sentences = re.split(r"(?<=[.!?])\s+", recent_text)
                                current_sentence = (
                                    sentences[-1] if sentences else recent_text
                                )

                                # Page/section explicit hint matching
                                page_hint = None
                                section_hint = None
                                page_match = re.search(
                                    r"page\s+(\d+)", current_sentence, re.IGNORECASE
                                )
                                if page_match:
                                    page_hint = int(page_match.group(1))
                                section_match = re.search(
                                    r"section\s+([\w\-\.\(\)]+)",
                                    current_sentence,
                                    re.IGNORECASE,
                                )
                                if section_match:
                                    section_hint = section_match.group(1)

                                selected_position = None

                                # Try pdfplumber positions first (more precise coordinates)
                                if has_pdfplumber:
                                    pdf_positions = pdf_text_blocks[conv_id]
                                    selected_position = find_best_match(
                                        current_sentence,
                                        pdf_positions,
                                        preferred_page=page_hint,
                                    )
                                    if selected_position:
                                        selected_position["source"] = "pdfplumber"

                                # Fall back to Snowflake document pages
                                if not selected_position and has_snowflake:
                                    selected_position = (
                                        find_best_match_in_snowflake_pages(
                                            current_sentence,
                                            conv_id,
                                            preferred_page=page_hint,
                                        )
                                    )

                                if conv_id:
                                    llm_referenced_quotes.setdefault(
                                        conv_id, []
                                    ).append((current_sentence, citation_num))

                                if selected_position:
                                    # Build extra details
                                    extra_details = {
                                        "Page": selected_position["page"],
                                        "Reference": (
                                            selected_position["text"][:100] + "..."
                                            if len(selected_position["text"]) > 100
                                            else selected_position["text"]
                                        ),
                                    }
                                    if section_hint:
                                        extra_details["Section"] = section_hint
                                    if selected_position.get(
                                        "source"
                                    ) == "snowflake" and selected_position.get(
                                        "file_name"
                                    ):
                                        extra_details["Document"] = selected_position[
                                            "file_name"
                                        ]

                                    # Create citation with the ACTUAL matching text
                                    citation_obj = cite(
                                        widget=widget,
                                        input_arguments=widget_input_args or {},
                                        extra_details=extra_details,
                                    )

                                    # Add bounding box for highlighting
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

                                    yield ("citation", citation_obj)

                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            f"[DEBUG] Created citation [{citation_num}] from "
                                            f"{selected_position.get('source', 'unknown')} source"
                                        )
                                else:
                                    # No match found in any source
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            "[DEBUG] No match found for citation in any source"
                                        )

                                    citation_obj = cite(
                                        widget=widget,
                                        input_arguments=widget_input_args or {},
                                    )
                                    yield ("citation", citation_obj)
                            else:
                                # No position data available at all
                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(
                                        f"[DEBUG] No position data for conv_id {conv_id}"
                                    )
                                citation_obj = cite(
                                    widget=widget,
                                    input_arguments=widget_input_args or {},
                                )
                                yield ("citation", citation_obj)

                        # Continue with rest of buffer
                        buffer = buffer[match.end() :]

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Stream error: {e}")
            import traceback

            traceback.print_exc()
        # Still yield what we have
        if buffer:
            yield ("text", buffer)
            buffer = ""

    # Yield any remaining buffer
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

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Final usage stats: {final_usage}")
        print(f"[DEBUG] Tool calls collected: {len(tool_calls)}")

    yield (
        "complete",
        {"text": full_text, "tool_calls": tool_calls, "usage": final_usage},
    )


async def generate_sse_events(stream_generator, stream_state: dict):
    """Consume the stream generator and yield SSE events."""
    from openbb_ai import citations, message_chunk
    from .helpers import to_sse

    try:
        async for event_type, data in stream_generator:
            if event_type == "text" and isinstance(data, str):
                yield to_sse(message_chunk(data))
                stream_state["full_text"] += data
            elif event_type == "citation":
                if hasattr(data, "extra_details") and data.extra_details:
                    page_num = data.extra_details.get("Page")
                    section = data.extra_details.get("Section")
                    if page_num:
                        prefix = f" (Page {page_num}"
                        if section:
                            prefix += f", Section {section}"
                        prefix += ")"
                        yield to_sse(message_chunk(prefix))
                        stream_state["full_text"] += prefix
                yield citations([data]).model_dump()
            elif event_type == "complete":
                if isinstance(data, dict):
                    stream_state["full_text"] = data.get("text", "")
                    stream_state["tool_calls"] = data.get("tool_calls", [])
                    stream_state["usage"] = data.get("usage")  # Capture usage stats
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Captured usage in stream_state: {stream_state.get('usage')}"
                        )
                else:
                    stream_state["full_text"] = str(data)
                    stream_state["tool_calls"] = []
    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] SSE generator error: {e}")
            import traceback

            traceback.print_exc()
        # Yield error message
        yield to_sse(message_chunk(f"Stream error: {str(e)}"))
