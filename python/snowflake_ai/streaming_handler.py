"""LLM streaming response handler for Snowflake AI."""

import asyncio
import os  # Move this to top-level import
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
    from .helpers import pdf_text_blocks, find_best_match, llm_referenced_quotes

    try:
        response_stream = await run_in_thread(
            client.chat_stream,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
        )
    except RuntimeError as e:
        if "tool calling is not supported" in str(e):
            response_stream = await run_in_thread(
                client.chat_stream,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=None,
            )
        else:
            raise

    full_text = ""
    tool_calls_in_progress = {}
    current_tool_use_id = None
    buffer = ""
    citations_created = {}  # Track which citation numbers we've created
    final_usage = None  # Track usage stats

    try:
        async for chunk in iterate_sync_generator(response_stream):
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[RAW SSE DATA]: {chunk}")

            # Check for usage data in chunk
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = chunk.usage
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Found usage in chunk: {usage_data}")

                # Extract usage fields - handle both dict and object formats
                if isinstance(usage_data, dict):
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

                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Extracted usage: {final_usage}")

            if not (hasattr(chunk, "choices") and chunk.choices):
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            delta_type = getattr(delta, "delta_type", None) or getattr(
                delta, "type", None
            )

            if delta_type == "tool_use":
                tool_use_id = getattr(delta, "tool_use_id", None) or getattr(
                    delta, "toolUseId", None
                )
                if tool_use_id:
                    current_tool_use_id = tool_use_id
                    if current_tool_use_id not in tool_calls_in_progress:
                        tool_calls_in_progress[current_tool_use_id] = {
                            "id": current_tool_use_id,
                            "name": getattr(delta, "name", None),
                            "arguments": "",
                        }

                input_piece = (
                    getattr(delta, "input", None)
                    or getattr(delta, "content", None)
                    or getattr(delta, "text", None)
                )
                if input_piece and current_tool_use_id:
                    tool_calls_in_progress[current_tool_use_id][
                        "arguments"
                    ] += input_piece

            elif delta_type == "text" or delta_type == "message":
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

                        # Create and yield citation if we have widget info and haven't created this one yet
                        if widget and citation_num not in citations_created:
                            citations_created[citation_num] = True

                            # Get PDF positions
                            if conv_id in pdf_text_blocks:
                                pdf_positions = pdf_text_blocks[conv_id]

                                # Get what the LLM JUST WROTE before this citation
                                # This is the ACTUAL CLAIM the LLM is making!
                                recent_text = (full_text + text_before).strip()

                                # Get the last sentence - this is what the citation refers to!
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

                                selected_position = find_best_match(
                                    current_sentence,
                                    pdf_positions,
                                    preferred_page=page_hint,
                                )

                                if conv_id:
                                    llm_referenced_quotes.setdefault(
                                        conv_id, []
                                    ).append((current_sentence, citation_num))

                                if selected_position:
                                    # Create citation with the ACTUAL matching text
                                    citation_obj = cite(
                                        widget=widget,
                                        input_arguments=widget_input_args or {},
                                        extra_details={
                                            "Page": selected_position["page"],
                                            "Reference": (
                                                selected_position["text"][:100] + "..."
                                                if len(selected_position["text"]) > 100
                                                else selected_position["text"]
                                            ),
                                            **(
                                                {"Section": section_hint}
                                                if section_hint
                                                else {}
                                            ),
                                        },
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
                                else:
                                    # No match found
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(f"[DEBUG] No match found for citation")

                                    # Don't highlight random shit!
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

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Final usage stats: {final_usage}")

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
