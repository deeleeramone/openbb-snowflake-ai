"""Helper functions for processing queries in the Snowflake AI server."""

import asyncio
import json
import os
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from openbb_ai import get_widget_data, reasoning_step
from openbb_ai.models import WidgetRequest

from .conversation_manager import format_messages_for_llm
from .document_processor import DocumentProcessor
from .helpers import (
    format_tool_overview,
    run_in_thread,
    seed_message_signatures,
    should_store_message,
    to_sse,
)
from .logger import get_logger
from .streaming_handler import (
    citations,
    generate_sse_events,
    message_chunk,
    stream_llm_with_tools,
)
from .tool_executor import execute_tool
from .widget_handler import WidgetHandler
from ._snowflake_ai import SnowflakeAI, ToolCall, FunctionCall

logger = get_logger(__name__)

# Session expiration error codes from Snowflake
SESSION_EXPIRED_CODES = {"390112", "390114", "390111"}


def is_session_expired_error(error: Exception) -> bool:
    """Check if an error indicates Snowflake session expiration."""
    error_str = str(error)
    return any(code in error_str for code in SESSION_EXPIRED_CODES)


async def handle_primary_widgets(
    request: Any,
    conv_id: str,
    client: SnowflakeAI,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Handle primary widgets explicitly added by the user.
    Returns a dictionary with widget context if successful, or yields SSE events.
    """
    last_message = request.messages[-1] if request.messages else None

    has_primary_widgets = (
        request.widgets and request.widgets.primary and len(request.widgets.primary) > 0
    )

    result_context = {
        "widget_context_str": "",
        "widget_for_citations": None,
        "widget_input_args_for_citations": None,
        "widget_context_metadata": None,
        "selected_widget_stage_path": None,
        "should_return": False,
    }

    if last_message and last_message.role in ["human", "user"] and has_primary_widgets:
        # Emit reasoning step BEFORE document processing starts
        yield to_sse(
            reasoning_step(
                "Reading document and metadata...",
                event_type="INFO",
            )
        )

        doc_proc = DocumentProcessor.instance()

        # Prepare document widgets (handles all document-specific logic)
        # Only pass primary widgets - secondary are not explicitly added by user
        doc_result = await doc_proc.prepare_document_widgets(
            request.widgets.primary,
            None,  # Don't use secondary widgets
            conv_id,
            client,
        )

        if doc_result["is_document"]:
            # Document widget found and processed
            result_context["widget_context_str"] = doc_result["widget_context_str"]
            result_context["widget_for_citations"] = doc_result["widget_for_citations"]
            result_context["widget_input_args_for_citations"] = doc_result[
                "widget_input_args_for_citations"
            ]
            result_context["widget_context_metadata"] = doc_result[
                "widget_context_metadata"
            ]
            result_context["selected_widget_stage_path"] = doc_result["stage_path"]
        else:
            # If document not ready or not found, proceed with original widget logic
            # Only call get_widget_data if there are actual widgets to request
            widget_requests = []
            for widget in request.widgets.primary or []:
                widget_req = WidgetRequest(
                    widget=widget,
                    input_arguments=(
                        {p.name: p.current_value for p in widget.params}
                        if hasattr(widget, "params")
                        else {}
                    ),
                )
                widget_requests.append(widget_req)

            # Only yield get_widget_data if there are actual requests
            if widget_requests:
                yield to_sse(
                    reasoning_step(
                        f"Calling tool, get_widget_data, with arguments -> {{'widget_requests': [{', '.join([str(req.widget.uuid) for req in widget_requests])}]}}",
                        event_type="INFO",
                    )
                )
                result = get_widget_data(widget_requests)
                yield result.model_dump()
                # Must return immediately after yielding get_widget_data to close the connection
                result_context["should_return"] = True
                yield result_context
                return

    yield result_context


async def load_conversation_history(
    client: SnowflakeAI,
    conv_id: str,
    refresh_client_func: Any,
) -> List[Dict[str, Any]]:
    """Load conversation history from cache."""
    try:
        cached_messages = await run_in_thread(client.get_messages, conv_id)
    except Exception as e:
        if is_session_expired_error(e):
            logger.warning("Session expired, refreshing client...")
            client = refresh_client_func(conv_id)
            cached_messages = await run_in_thread(client.get_messages, conv_id)
        else:
            raise

    # Build complete conversation history INCLUDING tool results
    all_messages = []
    for msg_id, role, content in cached_messages:
        # Check if this is a tool result message
        is_tool_result = "[Tool Result" in content
        all_messages.append(
            {
                "role": role,
                "content": content,
                "details": (
                    {"is_tool_result": is_tool_result} if is_tool_result else None
                ),
            }
        )

    # Prime the deduplication cache with existing history
    seed_message_signatures(conv_id, all_messages)

    # DEBUG: Verify no artifacts are being reconstructed from conversation history
    logger.debug(
        "[CONV HISTORY] Loaded %d messages from database for conv_id=%s",
        len(all_messages),
        conv_id,
    )
    logger.debug(
        "[CONV HISTORY] No artifacts reconstructed - conversation loading complete"
    )

    return all_messages


async def process_incoming_messages(
    request: Any,
    all_messages: List[Dict[str, Any]],
    conv_id: str,
    client: SnowflakeAI,
    selected_widget_stage_path: Optional[str],
    existing_widget_context_str: str = "",
    existing_widget_context_metadata: Optional[Dict[str, Any]] = None,
    existing_widget_for_citations: Optional[Any] = None,
    existing_widget_input_args: Optional[Dict[str, Any]] = None,
) -> Tuple[
    List[Dict[str, Any]],
    bool,
    bool,
    Optional[Any],
    Optional[Dict[str, Any]],
    str,
    Optional[Dict[str, Any]],
]:
    """
    Process incoming messages from the request, handle tool results, and update history.
    Returns updated all_messages, has_new_user_message, needs_response, widget_for_citations,
    widget_input_args_for_citations, widget_context_str, widget_context_metadata

    Args:
        existing_widget_context_str: Context string from primary widget processing (preserve if set)
        existing_widget_context_metadata: Metadata from primary widget processing
        existing_widget_for_citations: Widget for citations from primary widget processing
        existing_widget_input_args: Widget input args from primary widget processing
    """
    request_messages_to_add = []
    has_new_user_message = False
    needs_response = False

    # Preserve existing widget context from handle_primary_widgets, only overwrite if tool message provides new data
    widget_for_citations = existing_widget_for_citations
    widget_input_args_for_citations = existing_widget_input_args
    widget_context_str = existing_widget_context_str
    widget_context_metadata = existing_widget_context_metadata

    all_widgets: list[Any] = []
    if getattr(request, "widgets", None):
        all_widgets = list(request.widgets.primary or []) + list(
            request.widgets.secondary or []
        )

    def find_widget_by_uuid(target_uuid: str | None):
        if not target_uuid:
            return None
        for widget in all_widgets:
            if str(widget.uuid) == target_uuid:
                return widget
        return None

    for idx, message in enumerate(request.messages):
        # Handle tool messages (widget data comes back as tool messages)
        if message.role == "tool":
            # Only process if it has data
            if hasattr(message, "data") and message.data:
                message_input_args = getattr(message, "input_arguments", None)
                if not isinstance(message_input_args, dict):
                    message_input_args = None
                data_sources = (message_input_args or {}).get("data_sources", []) or []

                # Get the target widget to determine processing strategy
                target_widget = None
                known_filename = None
                if message_input_args and data_sources and all_widgets:
                    target_widget = find_widget_by_uuid(
                        data_sources[0].get("widget_uuid")
                    )

                # Process based on widget type
                widget_handler = await WidgetHandler.instance()

                if target_widget and widget_handler.is_document_widget(target_widget):
                    doc_proc = DocumentProcessor.instance()
                    known_filename = doc_proc.extract_filename_from_widget(
                        target_widget
                    )

                    parsed_data = widget_handler.parse_widget_data(
                        message.data, conv_id, client, known_filename
                    )
                    doc_proc.trigger_snowflake_upload_for_widget_pdf(client, conv_id)
                else:
                    parsed_data = message.data
                    if target_widget:
                        widget_result = await widget_handler.process_widget_response(
                            target_widget, message.data, conv_id, client
                        )
                        parsed_data = widget_result.get("data", message.data)

                # Store tabular/JSON widget data in Snowflake for reference
                if message_input_args and data_sources:
                    for data_source in data_sources:
                        widget_uuid = data_source.get("widget_uuid")
                        if widget_uuid and request.widgets:
                            # Find widget name
                            widget_name = widget_uuid
                            target_widget = find_widget_by_uuid(widget_uuid)
                            if target_widget:
                                widget_name = getattr(
                                    target_widget, "name", widget_uuid
                                )

                            # Store widget data in Snowflake (async, non-blocking)
                            doc_proc = DocumentProcessor.instance()
                            asyncio.create_task(
                                doc_proc.store_widget_data_in_snowflake(
                                    client=client,
                                    widget_uuid=widget_uuid,
                                    widget_name=widget_name,
                                    data_content=parsed_data,
                                    conversation_id=conv_id,
                                    data_type="json",
                                )
                            )

                # Only add context if it's the last message
                if idx == len(request.messages) - 1:
                    widget_data_request = data_sources[0] if data_sources else {}
                    target_uuid = (
                        widget_data_request.get("widget_uuid")
                        if isinstance(widget_data_request, dict)
                        else None
                    )
                    target_widget = find_widget_by_uuid(target_uuid)
                    widget_display_name = None
                    widget_description = None
                    widget_type = None
                    document_label = None
                    widget_label = None

                    if target_widget:
                        widget_display_name = getattr(
                            target_widget, "name", None
                        ) or getattr(target_widget, "title", None)
                        widget_description = getattr(target_widget, "description", None)
                        widget_type = getattr(target_widget, "type", None) or getattr(
                            target_widget, "kind", None
                        )

                    widget_input_args_dict = None
                    if isinstance(widget_data_request, dict):
                        widget_input_args_dict = dict(
                            widget_data_request.get("input_args", {}) or {}
                        )
                        widget_input_args_dict.setdefault("conversation_id", conv_id)

                    document_label = (
                        known_filename
                        or (widget_input_args_dict or {}).get("file_name")
                        or (widget_input_args_dict or {}).get("document_name")
                    )
                    if not document_label:
                        for candidate_key in (
                            "dataset_name",
                            "table_name",
                            "sheet_name",
                            "source_name",
                        ):
                            candidate_value = (widget_input_args_dict or {}).get(
                                candidate_key
                            )
                            if candidate_value:
                                document_label = candidate_value
                                break

                    stage_path = (
                        (widget_input_args_dict or {}).get("stage_path")
                        or (widget_data_request or {}).get("stage_path")
                        or selected_widget_stage_path
                    )
                    if widget_input_args_dict is not None and stage_path:
                        widget_input_args_dict.setdefault("stage_path", stage_path)

                    widget_type = (
                        widget_type
                        or (widget_data_request or {}).get("widget_type")
                        or (widget_data_request or {}).get("widget_kind")
                    )
                    widget_display_name = (
                        widget_display_name
                        or (widget_data_request or {}).get("widget_name")
                        or (widget_data_request or {}).get("widget_title")
                    )
                    widget_description = (
                        widget_description
                        or (widget_data_request or {}).get("description")
                        or (widget_input_args_dict or {}).get("description")
                    )

                    widget_label = (
                        widget_display_name or document_label or target_uuid or "widget"
                    )
                    if widget_input_args_dict is not None:
                        widget_input_args_dict.setdefault("widget_label", widget_label)
                        if target_widget:
                            widget_input_args_dict.setdefault(
                                "widget_uuid", str(target_widget.uuid)
                            )
                        widget_input_args_dict.setdefault(
                            "widget_title", widget_display_name
                        )
                        widget_input_args_for_citations = widget_input_args_dict

                    metadata_lines = [
                        "The user is explicitly referring to this widget data. Do NOT ask which widget or document; cite this source directly."
                    ]
                    if widget_display_name:
                        metadata_lines.append(f"Widget Name: {widget_display_name}")
                    if target_uuid:
                        metadata_lines.append(f"Widget UUID: {target_uuid}")
                    if document_label:
                        metadata_lines.append(f"Document/File: {document_label}")
                    if stage_path:
                        metadata_lines.append(f"Stage Path: {stage_path}")
                    if widget_type:
                        metadata_lines.append(f"Widget Type: {widget_type}")
                    if widget_description:
                        metadata_lines.append(
                            f"Widget Description: {widget_description}"
                        )

                    data_label = widget_display_name or document_label or "Widget Data"
                    widget_context_str = (
                        "\\n".join(metadata_lines)
                        + f"\\n\\n--- Widget Data: {data_label} ---\\n{parsed_data}\\n------\\n"
                    )

                    # Extract widget info for citations
                    if target_widget:
                        widget_for_citations = target_widget

                    widget_context_metadata = {
                        "widget_label": widget_label,
                        "widget_uuid": target_uuid,
                        "document_label": document_label,
                        "stage_path": stage_path,
                    }
            # Skip adding tool message to current_messages
            continue

        elif hasattr(message, "content") and message.content:
            # Check if this is a new message not in cache
            is_new = True
            if all_messages:
                message_content = (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                )

                # Prevent duplicates by checking entire content
                for cached_msg in all_messages:
                    if (
                        cached_msg["role"] == message.role
                        and cached_msg["content"] == message_content
                    ):
                        is_new = False
                        break

            if is_new:
                request_messages_to_add.append(message)
                # Track if we have a new user message
                if message.role in ["human", "user"]:
                    has_new_user_message = True
                    needs_response = True

            else:
                # Even if message is cached, check if it's the last user message
                # and whether it has been responded to
                if (
                    message.role in ["human", "user"]
                    and idx == len(request.messages) - 1
                ):
                    # Treat a last human message as an intentional send/resend.
                    # Always require a fresh response when the user explicitly sent (or resent) the message.
                    has_new_user_message = True
                    needs_response = True
                    logger.debug("Last user message is resend - forcing fresh response")

                    # Note: we intentionally do NOT append the duplicate to request_messages_to_add
                    # to avoid duplicating stored messages in the cache.
                else:
                    has_response = False
                    found_this_msg = False

                    for i, cached_msg in enumerate(all_messages):
                        if not found_this_msg:
                            # Find this specific user message in cache
                            if cached_msg["role"] in [
                                "human",
                                "user",
                            ] and cached_msg["content"] == (
                                message.content
                                if isinstance(message.content, str)
                                else str(message.content)
                            ):
                                found_this_msg = True
                        elif found_this_msg:
                            # After finding the user message, check if there's an assistant response
                            if cached_msg["role"] == "assistant":
                                has_response = True
                                break
                            elif (
                                cached_msg["role"] in ["human", "user"]
                                and "[Tool Result" not in cached_msg["content"]
                            ):
                                # Another user message without tool result means no response to previous
                                break

                    if not has_response:
                        needs_response = True
                        logger.debug("Last user message needs a response")

    if widget_for_citations:
        if widget_input_args_for_citations is None:
            widget_input_args_for_citations = {"conversation_id": conv_id}
        else:
            widget_input_args_for_citations.setdefault("conversation_id", conv_id)

    # Store new messages FIRST before any early returns
    # Only add truly new unique messages from request and store them
    if request_messages_to_add:
        for message in request_messages_to_add:
            msg_dict = {
                "role": message.role,
                "content": (
                    message.content
                    if isinstance(message.content, str)
                    else str(message.content)
                ),
                "details": None,
            }
            if should_store_message(
                conv_id,
                msg_dict["role"],
                msg_dict["content"],
                details=msg_dict["details"],
            ):
                msg_id = str(uuid.uuid4())
                all_messages.append(msg_dict)

                # Store new messages in Snowflake
                await run_in_thread(
                    client.add_message,
                    conv_id,
                    msg_id,
                    msg_dict["role"],
                    msg_dict["content"],
                )

    return (
        all_messages,
        has_new_user_message,
        needs_response,
        widget_for_citations,
        widget_input_args_for_citations,
        widget_context_str,
        widget_context_metadata,
    )


async def prepare_llm_context(
    request: Any,
    all_messages: List[Dict[str, Any]],
    widget_context_str: str,
    widget_context_metadata: Optional[Dict[str, Any]],
    conv_id: str,
    client: SnowflakeAI,
    selected_model: str,
    supports_tools: bool,
    tools: Optional[List[Any]],
) -> Tuple[List[Tuple[str, str]], Optional[str]]:
    """
    Prepare the context for the LLM, including system prompt and message formatting.
    Returns formatted messages and tool overview.
    """
    # Get the CURRENT user message from the request (not from cached history)
    current_request_user_msg = None
    for msg in reversed(request.messages):
        if msg.role in ["human", "user"]:
            content = getattr(msg, "content", None)
            if content:
                current_request_user_msg = (
                    content if isinstance(content, str) else str(content)
                )
            break

    # Keep a sliding window of recent messages, but the full history is available if needed
    current_messages = []

    # Determine if we need full history based on the user's query
    # Use the CURRENT request message, not old cached history
    last_user_msg_raw = current_request_user_msg or ""
    last_user_msg_lower = last_user_msg_raw.lower()

    # Check if user is asking about conversation history or previous data
    # Expanded keywords to catch more cases where full history is needed
    needs_full_history = any(
        phrase in last_user_msg_lower
        for phrase in [
            "earlier",
            "previous",
            "history",
            "conversation",
            "what did",
            "what was",
            "what is",
            "you said",
            "we discussed",
            "remember",
            "recall",
            "mentioned",
            "show me again",
            "repeat",
            "before",
            "ago",
            "extract",
            "table",
            "data from",
            "message",
            "context",
            "cached",
            "stored",
            "available",
            "access",
            "tool output",
            "last active",
            "reassess",
            "situation",
            "improved",
            "context map",
            "what tool",
        ]
    )

    if needs_full_history:
        # User is asking about history - include ALL messages
        MAX_LLM_CONTEXT_MESSAGES = 200  # Increased to ensure we get everything
        logger.debug(
            "User query references history - including up to %d messages",
            MAX_LLM_CONTEXT_MESSAGES,
        )  # Include all messages for history queries
        current_messages = all_messages[-MAX_LLM_CONTEXT_MESSAGES:]
    else:
        # Normal query - use sliding window for efficiency
        SLIDING_WINDOW_SIZE = 30
        MAX_TOOL_RESULTS_FOR_CONTEXT = 10
        all_tool_results = []
        for msg in all_messages:
            if msg["role"] in ["user"] and "[Tool Result" in msg["content"]:
                all_tool_results.append(msg)

        if len(all_tool_results) > MAX_TOOL_RESULTS_FOR_CONTEXT:
            all_tool_results = all_tool_results[-MAX_TOOL_RESULTS_FOR_CONTEXT:]

        # Get recent conversation messages
        recent_conversation = []
        for msg in all_messages[-(SLIDING_WINDOW_SIZE):]:
            if msg not in all_tool_results:
                recent_conversation.append(msg)

        # Combine: ALL tool results + recent conversation
        current_messages = all_tool_results + recent_conversation

        # Sort by original order
        current_messages = sorted(
            current_messages,
            key=lambda x: all_messages.index(x) if x in all_messages else 0,
        )

    # Rebuild the current user turn to avoid contaminating future turns with old context
    enriched_user_message = None
    if current_request_user_msg:
        enriched_user_message = current_request_user_msg
    elif widget_context_str:
        # No explicit user text but widget data exists (e.g., tool follow-up)
        enriched_user_message = "[User Context inferred from widget selection]"

    if enriched_user_message is not None:
        if widget_context_str:
            enriched_user_message += "\\n\\n" + widget_context_str
        elif widget_context_metadata:
            # Widget selected but no full context string built yet - inject explicit selection
            doc_name = widget_context_metadata.get("document_label", "unknown")
            stage = widget_context_metadata.get("stage_path", "")
            enriched_user_message += f"\\n\\nWidget provided document: {doc_name}"
            if stage:
                enriched_user_message += f" ({stage})"
    elif widget_context_metadata:
        # No user message at all but widget is selected
        doc_name = widget_context_metadata.get("document_label", "unknown")
        stage = widget_context_metadata.get("stage_path", "")
        enriched_user_message = f"Widget provided document: {doc_name}"
        if stage:
            enriched_user_message += f" ({stage})"
        enriched_user_message += "\\n\\nPlease analyze this document."

        # Remove any trailing user/human messages that mirror this content to prevent duplication
        while current_messages and current_messages[-1]["role"] in [
            "human",
            "user",
        ]:
            last_content = current_messages[-1].get("content", "")
            if last_content.strip() == enriched_user_message.strip() or (
                current_request_user_msg
                and last_content.strip() == current_request_user_msg.strip()
            ):
                current_messages.pop()
            else:
                break

        current_messages.append(
            {
                "role": "user",
                "content": enriched_user_message,
                "details": None,
            }
        )

    # Get user schema for document storage - ALWAYS OPENBB_AGENTS.USER_{username}
    snowflake_user = await run_in_thread(client.get_current_user)
    sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
    user_schema = f"USER_{sanitized_user}".upper()

    # Validate message structure when using tools
    if supports_tools and tools and current_messages:
        # Remove trailing assistant messages when using tools
        while current_messages and current_messages[-1]["role"] in (
            "assistant",
            "ai",
        ):
            current_messages.pop()

        # Also ensure we're not sending tool results as the last message
        while current_messages and "[Tool Result" in current_messages[-1].get(
            "content", ""
        ):
            current_messages.pop()

        # If we removed everything, we need to ensure there's at least the latest user message
        if not current_messages or current_messages[-1]["role"] not in [
            "human",
            "user",
        ]:
            # Use the CURRENT request's user message, not old history
            if current_request_user_msg:
                current_messages.append(
                    {
                        "role": "user",
                        "content": current_request_user_msg,
                        "details": None,
                    }
                )
            else:
                # Fallback to finding from all_messages
                for msg in reversed(all_messages):
                    if msg["role"] in [
                        "human",
                        "user",
                    ] and "[Tool Result" not in msg.get("content", ""):
                        current_messages.append(msg)
                        break

        # CRITICAL: Ensure the CURRENT request message is the last user message
        # This handles the case where user resends a previous question
        if current_request_user_msg:
            last_msg = current_messages[-1] if current_messages else None
            if not last_msg or last_msg.get("content") != current_request_user_msg:
                # The current request message is different from the last message
                # Remove any trailing messages that come after what should be answered
                # and ensure current request is last
                current_messages.append(
                    {
                        "role": "user",
                        "content": current_request_user_msg,
                        "details": None,
                    }
                )

    tool_overview = format_tool_overview(tools)

    from .system_prompt import build_system_prompt

    # Run all these Snowflake queries in PARALLEL to reduce latency
    async def get_docs():
        try:
            return await run_in_thread(client.list_cortex_documents)
        except Exception:
            return None

    async def get_db():
        try:
            return await run_in_thread(client.get_current_database)
        except Exception:
            return None

    async def get_schema():
        try:
            return await run_in_thread(client.get_current_schema)
        except Exception:
            return None

    # Execute in parallel
    available_docs, current_database, current_schema = await asyncio.gather(
        get_docs(),
        get_db(),
        get_schema(),
    )

    # Get document structure if available for this conversation
    doc_proc = DocumentProcessor.instance()
    document_structure = doc_proc.format_document_structure_for_llm(conv_id)

    system_prompt = build_system_prompt(
        total_messages=len(all_messages),
        current_messages=len(current_messages),
        user_schema=user_schema,
        current_database=current_database,
        current_schema=current_schema,
        widget_context_metadata=widget_context_metadata,
        available_docs=available_docs,
        tool_overview=(tool_overview if supports_tools and tools else None),
        supports_tools=supports_tools and bool(tools),
        document_structure=document_structure,
    )

    # Prepare the final message stream
    ai_messages_formatted_tuples = format_messages_for_llm(
        current_messages,
        system_prompt,
        inject_widget_data=False,
    )

    if supports_tools and tools and ai_messages_formatted_tuples:
        while (
            ai_messages_formatted_tuples
            and ai_messages_formatted_tuples[-1][0] == "assistant"
        ):
            ai_messages_formatted_tuples.pop()

        # Ensure we still have messages after cleanup
        if (
            not ai_messages_formatted_tuples
            or ai_messages_formatted_tuples[-1][0] == "assistant"
        ):
            # Find the last user message and ensure it's in the list
            for msg in reversed(current_messages):
                if msg["role"] in [
                    "human",
                    "user",
                ] and "[Tool Result" not in msg.get("content", ""):
                    ai_messages_formatted_tuples.append(("user", msg["content"]))
                    break

    return ai_messages_formatted_tuples, tool_overview


async def stream_response_no_tools(
    client: SnowflakeAI,
    ai_messages_formatted_tuples: List[Tuple[str, str]],
    selected_model: str,
    selected_temperature: float,
    selected_max_tokens: int,
    conv_id: str,
    widget_for_citations: Any,
    widget_input_args_for_citations: Any,
    token_usage: Dict[str, Dict[str, int]],
    all_messages: List[Dict[str, Any]],
) -> AsyncGenerator[Dict[str, Any], None]:
    """Stream response for models that don't support tool calling."""
    stream_state = {
        "full_text": "",
        "tool_calls": [],
        "usage": None,
        "citation_count": 0,
        "citation_summaries": [],
        "fatal_error": None,
    }

    generator = stream_llm_with_tools(
        client,
        ai_messages_formatted_tuples,
        selected_model,
        selected_temperature,
        selected_max_tokens,
        tools=None,
        conv_id=conv_id,
        widget=widget_for_citations,
        widget_input_args=widget_input_args_for_citations,
    )

    # Iterate with timeout to prevent indefinite hangs
    sse_timeout = float(os.environ.get("SSE_EVENT_TIMEOUT", "120"))
    try:
        async for event in generate_sse_events(generator, stream_state):
            yield event
            # Check if stream marked as failed
            if stream_state.get("fatal_error"):
                break
    except asyncio.TimeoutError:
        logger.error("SSE stream timed out after %s seconds", sse_timeout)
        stream_state["fatal_error"] = f"Stream timeout after {sse_timeout}s"
        yield to_sse(message_chunk(f"❌ Stream timed out after {sse_timeout}s"))

    # Update token usage after completion
    if stream_state.get("usage"):
        if conv_id not in token_usage:
            token_usage[conv_id] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_requests": 0,
            }

        usage = stream_state["usage"]
        token_usage[conv_id]["prompt_tokens"] += usage.get("prompt_tokens", 0)
        token_usage[conv_id]["completion_tokens"] += usage.get("completion_tokens", 0)
        token_usage[conv_id]["total_tokens"] += usage.get("total_tokens", 0)
        token_usage[conv_id]["api_requests"] += 1

        # Store in cache
        try:
            await run_in_thread(
                client.set_conversation_data,
                conv_id,
                "token_usage",
                json.dumps(token_usage[conv_id]),
            )
        except Exception as e:
            logger.error("Non-tool path - ERROR storing token_usage: %s", e)

    if stream_state["full_text"].strip() and not stream_state.get("fatal_error"):
        # Check if this assistant response is already in cache
        response_already_cached = False
        for cached_msg in all_messages[-5:]:  # Check last few messages
            if (
                cached_msg["role"] == "assistant"
                and cached_msg["content"] == stream_state["full_text"]
            ):
                response_already_cached = True
                break

        if not response_already_cached:
            assistant_entry = {
                "role": "assistant",
                "content": stream_state["full_text"],
                "details": {"message_type": "assistant_final"},
            }
            if should_store_message(
                conv_id,
                assistant_entry["role"],
                assistant_entry["content"],
                details=assistant_entry["details"],
            ):
                ai_msg_id = str(uuid.uuid4())
                all_messages.append(assistant_entry)
                await run_in_thread(
                    client.add_message,
                    conv_id,
                    ai_msg_id,
                    assistant_entry["role"],
                    assistant_entry["content"],
                )


async def execute_single_tool(
    tool_call: Any,
    client: SnowflakeAI,
    conv_id: str,
    request: Any,
    all_messages: List[Dict[str, Any]],
    ai_messages_formatted_tuples: List[Tuple[str, str]],
    text2sql_will_auto_execute: bool,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Execute a single tool call and handle its output."""
    tool_name = tool_call.function.name
    tool_args_str = tool_call.function.arguments

    # Skip execute_query if text2sql is in the batch - it will auto-execute
    if tool_name == "execute_query" and text2sql_will_auto_execute:
        return

    # Yield reasoning step with arguments in standard format
    yield to_sse(
        reasoning_step(
            f"Calling tool, {tool_name}, with arguments -> {tool_args_str}",
            event_type="INFO",
        )
    )

    # Execute the tool
    # INTERCEPT get_widget_data tool calls
    if tool_name == "get_widget_data":
        doc_proc = DocumentProcessor.instance()

        try:
            tool_args = json.loads(tool_args_str) if tool_args_str else {}
        except json.JSONDecodeError:
            tool_args = {}

        current_tool_output_for_llm, raw_tool_data = (
            await doc_proc.handle_get_widget_data_tool_call(
                tool_args=tool_args,
                widgets_primary=(request.widgets.primary if request.widgets else None),
                widgets_secondary=(
                    request.widgets.secondary if request.widgets else None
                ),
                client=client,
                conversation_id=conv_id,
            )
        )

        # If we got a widget_data_request, yield it for the UI to fetch
        if isinstance(raw_tool_data, dict) and "widget_data_request" in raw_tool_data:
            yield raw_tool_data["widget_data_request"]
            # The actual data will come back in a subsequent request
            # Skip normal tool result processing for this case
            return
    else:
        # This is a generator, so we need to iterate
        logger.debug(
            "[TOOL EXECUTE] About to execute tool: %s with args: %s",
            tool_name,
            tool_args_str[:200] if tool_args_str else "None",
        )

        current_tool_output_for_llm = ""
        raw_tool_data = None
        tool_output_generator = execute_tool(tool_call, client, conv_id)
        try:
            async for event in tool_output_generator:
                # The generator will yield reasoning steps first, then the final result
                if isinstance(event, tuple) and len(event) == 2:
                    (
                        current_tool_output_for_llm,
                        raw_tool_data,
                    ) = event
                elif isinstance(event, dict) and event.get("event") == "reasoning_step":
                    yield event
                elif isinstance(event, dict) and "event" in event:
                    # SSE events like chart artifacts, tables, etc.
                    # These come from to_sse() wrapped objects
                    event_type = event.get("event", "unknown")
                    if event_type == "copilotMessageArtifact":
                        # Log artifact being yielded from server
                        data_preview = str(event.get("data", ""))[:300]
                        logger.debug(
                            "[SERVER ARTIFACT] Yielding artifact event. Data preview: %s",
                            data_preview,
                        )
                    yield event
                elif hasattr(event, "event") and hasattr(event, "data"):
                    # MessageArtifactSSE objects (charts, tables) from openbb_ai.helpers
                    sse_event = to_sse(event)
                    if sse_event.get("event") == "copilotMessageArtifact":
                        data_preview = str(sse_event.get("data", ""))[:300]
                        logger.debug(
                            "[SERVER ARTIFACT] Yielding converted artifact. Data preview: %s",
                            data_preview,
                        )
                    yield sse_event
                else:
                    # Log unexpected event types for debugging
                    logger.debug("Unhandled tool event type: %s", type(event))
        except Exception as tool_exc:  # pragma: no cover - defensive
            error_message = f"Error executing tool {tool_name}: {tool_exc}"
            logger.error(
                "Tool execution failure for %s: %s",
                tool_name,
                tool_exc,
                exc_info=True,
            )
            yield to_sse(
                reasoning_step(
                    error_message,
                    event_type="ERROR",
                )
            )
            current_tool_output_for_llm = error_message
            raw_tool_data = {"error": str(tool_exc)}

    if not current_tool_output_for_llm:
        fallback_msg = f"Error: Tool {tool_name} did not return any output."
        yield to_sse(
            reasoning_step(
                fallback_msg,
                event_type="WARNING",
            )
        )
        current_tool_output_for_llm = fallback_msg
        raw_tool_data = raw_tool_data or {"error": "empty_result"}

    # For text2sql tool: check if user explicitly asked for SQL code
    # If user asked for SQL output → return SQL and exit
    # If LLM is using text2sql to help generate SQL → automatically execute it
    text2sql_auto_executed = False
    if tool_name == "text2sql":
        sql_generated = False
        sql_text = ""
        if isinstance(raw_tool_data, dict):
            sql_text = raw_tool_data.get("sql", "")
            sql_generated = bool(sql_text and sql_text.strip())

        if sql_generated:
            # Check if user explicitly asked for SQL code output
            user_wants_sql_code = False
            last_user_msg = ""
            for msg in reversed(all_messages):
                if msg.get("role") in [
                    "user",
                    "human",
                ] and "[Tool Result" not in msg.get("content", ""):
                    last_user_msg = msg.get("content", "").lower()
                    break

            sql_output_keywords = [
                "write me a query",
                "write a query",
                "generate sql",
                "generate a query",
                "create a query",
                "show me the sql",
                "give me the sql",
                "what's the sql",
                "what is the sql",
                "sql for",
                "query for",
                "write sql",
            ]
            user_wants_sql_code = any(kw in last_user_msg for kw in sql_output_keywords)

            if user_wants_sql_code:
                # User explicitly asked for SQL code - return it and exit
                yield to_sse(message_chunk(current_tool_output_for_llm))
                tool_result_text = (
                    f"[Tool Result from {tool_name}]\n{current_tool_output_for_llm}"
                )
                tool_message = {
                    "role": "user",
                    "content": tool_result_text,
                    "details": {
                        "is_tool_result": True,
                        "message_type": "tool_result",
                        "tool_name": tool_name,
                    },
                }
                if should_store_message(
                    conv_id,
                    tool_message["role"],
                    tool_message["content"],
                    details=tool_message["details"],
                ):
                    tool_msg_id = str(uuid.uuid4())
                    all_messages.append(tool_message)
                    await run_in_thread(
                        client.add_message,
                        conv_id,
                        tool_msg_id,
                        tool_message["role"],
                        tool_message["content"],
                    )
                return
            else:
                # LLM is using text2sql to get SQL - automatically execute it
                yield to_sse(
                    reasoning_step(
                        f'Calling tool, execute_query, with arguments -> {{"query": "{sql_text}"}}',
                        event_type="INFO",
                    )
                )

                # Create execute_query tool call
                execute_tool_call = ToolCall(
                    id=str(uuid.uuid4()),
                    tool_type="function",
                    function=FunctionCall(
                        name="execute_query",
                        arguments=json.dumps({"query": sql_text}),
                    ),
                )

                # Execute the query
                exec_output = ""
                exec_raw_data = None
                exec_generator = execute_tool(execute_tool_call, client, conv_id)
                try:
                    async for exec_event in exec_generator:
                        if isinstance(exec_event, tuple) and len(exec_event) == 2:
                            exec_output, exec_raw_data = exec_event
                        elif isinstance(exec_event, dict) and "event" in exec_event:
                            # SSE events like table artifacts
                            event_type = exec_event.get("event", "unknown")
                            if event_type == "copilotMessageArtifact":
                                data_preview = str(exec_event.get("data", ""))[:300]
                                logger.debug(
                                    "[AUTO-EXEC ARTIFACT] Yielding artifact from text2sql auto-execute. Data preview: %s",
                                    data_preview,
                                )
                            yield exec_event
                        elif hasattr(exec_event, "event") and hasattr(
                            exec_event, "data"
                        ):
                            sse_event = to_sse(exec_event)
                            if sse_event.get("event") == "copilotMessageArtifact":
                                data_preview = str(sse_event.get("data", ""))[:300]
                                logger.debug(
                                    "[AUTO-EXEC ARTIFACT] Yielding converted artifact from text2sql auto-execute. Data preview: %s",
                                    data_preview,
                                )
                            yield sse_event
                except Exception as exec_exc:
                    exec_output = f"Query execution failed: {exec_exc}"
                    exec_raw_data = {"error": str(exec_exc)}

                yield to_sse(
                    reasoning_step(
                        "Tool execute_query completed successfully.",
                        event_type="INFO",
                    )
                )

                # Update the tool output to include the query results
                current_tool_output_for_llm = f"Generated SQL:\n```sql\n{sql_text}\n```\n\nQuery Results:\n{exec_output}"
                raw_tool_data = exec_raw_data
                text2sql_auto_executed = True

    # Yield completion reasoning step (skip if text2sql already auto-executed)
    if text2sql_auto_executed:
        # Already yielded execute_query completion, show row count if available
        if isinstance(raw_tool_data, dict) and "rowData" in raw_tool_data:
            row_count = len(raw_tool_data["rowData"])
            yield to_sse(
                reasoning_step(
                    f"Retrieved {row_count} rows.",
                    event_type="INFO",
                )
            )
    elif "Error getting" in str(current_tool_output_for_llm) or "Error:" in str(
        current_tool_output_for_llm
    ):
        yield to_sse(
            reasoning_step(
                f"Tool {tool_name} encountered an error.",
                event_type="ERROR",
            )
        )
    else:
        if (
            tool_name in ["execute_query", "get_table_sample_data"]
            and isinstance(raw_tool_data, dict)
            and "rowData" in raw_tool_data
        ):
            row_count = len(raw_tool_data["rowData"])
            yield to_sse(
                reasoning_step(
                    f"Retrieved {row_count} rows.",
                    event_type="INFO",
                )
            )
        else:
            yield to_sse(
                reasoning_step(
                    f"Tool {tool_name} completed successfully.",
                    event_type="INFO",
                )
            )

    # Format tool result for the LLM - apply intelligent compression
    tool_result_formatted = (
        f"The result from {tool_name} is:\n{current_tool_output_for_llm}"
    )

    # Add as a user message to continue conversation
    ai_messages_formatted_tuples.append(
        ("user", f"[Tool Result]\n{tool_result_formatted}")
    )

    # Store tool result in cache
    tool_result_text = f"[Tool Result from {tool_name}]\n{current_tool_output_for_llm}"
    tool_message = {
        "role": "user",
        "content": tool_result_text,
        "details": {
            "is_tool_result": True,
            "message_type": "tool_result",
            "tool_name": tool_name,
        },
    }

    if should_store_message(
        conv_id,
        tool_message["role"],
        tool_message["content"],
        details=tool_message["details"],
    ):
        tool_msg_id = str(uuid.uuid4())
        all_messages.append(tool_message)
        await run_in_thread(
            client.add_message,
            conv_id,
            tool_msg_id,
            tool_message["role"],
            tool_message["content"],
        )

        # Store raw data for direct access
        if isinstance(raw_tool_data, dict):
            data_key = f"tool_result_{tool_name}_{tool_msg_id}"
            await run_in_thread(
                client.set_conversation_data,
                conv_id,
                data_key,
                json.dumps(raw_tool_data),
            )
