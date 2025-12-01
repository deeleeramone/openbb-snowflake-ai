"""OpenBB AI Agent server for Snowflake AI_COMPLETE."""

# flake8: noqa: PLR0911, PLR0912, T201
# pylint: disable = R0911, R0912, R0914, R0915, R0917, C0103, C0415, E0611, W0718

import os
import asyncio
import json
import shutil
import uuid
import traceback
from pathlib import Path
from typing import Any, AsyncGenerator

from .logger import get_logger

logger = get_logger(__name__)

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb_ai import (
    QueryRequest,
    citations,
    message_chunk,
    reasoning_step,
)
from sse_starlette import EventSourceResponse

# These are the Rust bindings
from ._snowflake_ai import (
    SnowflakeAgent,
    SnowflakeAI,
    ToolCall,
    FunctionCall,
)
from .slash_commands import handle_slash_command
from .helpers import (
    seed_message_signatures,
    should_store_message,
    to_sse,
    run_in_thread,
)
from .widgets import router as widgets_router

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://127.0.0.1:8000")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(widgets_router)


# Connection pool indexed by conversation_id
agent_pool: dict[str, SnowflakeAgent] = {}
# Store model preferences per conversation
model_preferences: dict[str, str] = {}
# Store temperature preferences per conversation
temperature_preferences: dict[str, float] = {}
# Store max_tokens preferences per conversation
max_tokens_preferences: dict[str, int] = {}
# Snowflake Client pool.
client_pool: dict[str, SnowflakeAI] = {}
# Track token usage per conversation
token_usage: dict[str, dict[str, int]] = {}
MAX_TOOL_ITERATIONS = 10

NON_TOOL_CALLING_MODELS = {
    "llama4-maverick",
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    # "deepseek-r1",
    "mistral-7b",
    "mistral-large",
    "mistral-large2",
    "snowflake-llama-3.3-70b",
    # Add more models that don't support tool calling
}

# Session expiration error codes from Snowflake
SESSION_EXPIRED_CODES = {"390112", "390114", "390111"}


def is_session_expired_error(error: Exception) -> bool:
    """Check if an error indicates Snowflake session expiration."""
    error_str = str(error)
    return any(code in error_str for code in SESSION_EXPIRED_CODES)


def refresh_client(conversation_id: str) -> SnowflakeAI:
    """Force refresh the Snowflake client for a conversation."""
    logger.info("Refreshing Snowflake client for conversation %s", conversation_id)

    # Remove old client and agent
    if conversation_id in client_pool:
        try:
            client_pool[conversation_id].close()
        except Exception:
            pass
        del client_pool[conversation_id]

    if conversation_id in agent_pool:
        del agent_pool[conversation_id]

    # Create fresh client
    client = SnowflakeAI(
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or "",
        database=os.environ.get("SNOWFLAKE_DATABASE") or "",
        schema=os.environ.get("SNOWFLAKE_SCHEMA") or "",
    )
    client_pool[conversation_id] = client
    agent_pool[conversation_id] = client.create_agent()

    return client


def get_or_create_agent(conversation_id: str = "default") -> SnowflakeAgent:
    """Get or create a client for the conversation."""
    if conversation_id not in agent_pool:
        client = SnowflakeAI(
            user=os.environ.get("SNOWFLAKE_USER"),
            password=os.environ.get("SNOWFLAKE_PASSWORD"),
            account=os.environ.get("SNOWFLAKE_ACCOUNT"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or "",
            database=os.environ.get("SNOWFLAKE_DATABASE") or "",
            schema=os.environ.get("SNOWFLAKE_SCHEMA") or "",
        )
        client_pool[conversation_id] = client
        agent_pool[conversation_id] = client.create_agent()

        # Load preferences from cache if available
        try:
            cached_model = client.get_conversation_data(
                conversation_id, "model_preference"
            )
            if cached_model:
                model_preferences[conversation_id] = cached_model
            else:
                client.set_conversation_data(
                    conversation_id,
                    "model_preference",
                    model_preferences.get(conversation_id, "openai-gpt-5-chat"),
                )

            cached_temperature = client.get_conversation_data(
                conversation_id, "temperature_preference"
            )
            if cached_temperature:
                temperature_preferences[conversation_id] = float(cached_temperature)
            else:
                client.set_conversation_data(
                    conversation_id,
                    "temperature_preference",
                    str(temperature_preferences.get(conversation_id, 0.7)),
                )

            cached_max_tokens = client.get_conversation_data(
                conversation_id, "max_tokens_preference"
            )
            if cached_max_tokens:
                max_tokens_preferences[conversation_id] = int(cached_max_tokens)
            else:
                client.set_conversation_data(
                    conversation_id,
                    "max_tokens_preference",
                    str(max_tokens_preferences.get(conversation_id, 4096)),
                )

            # Initialize token usage for new conversation
            if conversation_id not in token_usage:
                token_usage[conversation_id] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "api_requests": 0,
                }

                # Try to load from cache
                try:
                    cached_usage = client.get_conversation_data(
                        conversation_id, "token_usage"
                    )
                    if cached_usage:
                        token_usage[conversation_id] = json.loads(cached_usage)
                except Exception:
                    pass

        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.debug(
                    "Error loading preferences from cache for %s: %s",
                    conversation_id,
                    e,
                )

    return agent_pool[conversation_id]


async def shutdown_event():
    """Close all active client connections on shutdown."""
    for client in client_pool.values():
        try:
            await run_in_thread(client.close)
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Error closing client connection: %s", e)


app.add_event_handler(event_type="shutdown", func=shutdown_event)


@app.get("/agents.json")
async def agents_json():
    """Agent metadata endpoint."""
    return JSONResponse(
        content={
            "snowflake-ai": {
                "name": "Snowflake AI",
                "description": """
                Ask me about the tools I have access to and, how I can help you analyze your Snowflake data.

                Use /help to get a list of available slash commands.
                """,
                "image": f"{AGENT_BASE_URL}/logo.png",
                "endpoints": {
                    "query": f"{AGENT_BASE_URL}/query",
                },
                "features": {
                    "streaming": True,
                    "widget-dashboard-select": True,
                    "widget-dashboard-search": False,
                },
            }
        }
    )


@app.get("/widgets.json", include_in_schema=False)
async def get_widgets() -> dict:
    """Endpoint to get the widget configuration."""
    widgets_path = Path(__file__).parent / "widgets.json"
    with open(widgets_path, encoding="utf-8") as file:
        widgets = json.load(file)
    return widgets


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Endpoint to upload a file to a Snowflake stage."""
    conversation_id = request.headers.get("x-trace-id") or "default"
    get_or_create_agent(conversation_id)
    client = client_pool[conversation_id]

    temp_dir = Path("/tmp/snowflake_ai_uploads")
    temp_dir.mkdir(exist_ok=True)

    file_path = temp_dir / (file.filename or "uploaded_file")

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        stage_name = "CORTEX_UPLOADS"

        stage_path = await run_in_thread(
            client.upload_file_to_stage, str(file_path), stage_name
        )

        return JSONResponse(
            content={
                "status": "success",
                "stage_path": stage_path,
                "filename": file.filename,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        if file_path.exists():
            file_path.unlink()


@app.post("/upload_image")
async def upload_image(conversation_id: str, file: UploadFile = File(...)):
    """Endpoint to upload and process an image for a conversation."""
    raise NotImplementedError("Image upload is not implemented yet.")


@app.post("/query")
async def stream(request_obj: Request, request: QueryRequest):
    """Query endpoint with SSE streaming."""

    # Extract conversation ID IMMEDIATELY when request comes in
    conv_id = request_obj.headers.get("x-trace-id") or "default"

    # Ensure agent and preferences are loaded from cache BEFORE reading them
    get_or_create_agent(conv_id)

    # Get all preferences IMMEDIATELY after we have conv_id
    selected_model = model_preferences.get(conv_id, "openai-gpt-5-chat")
    selected_temperature = temperature_preferences.get(conv_id, 0.7)
    selected_max_tokens = max_tokens_preferences.get(conv_id, 4096)

    async def execution_loop():
        """Main execution loop."""
        # Check for slash commands FIRST
        if request.messages and request.messages[-1].role in ["human", "user"]:
            last_message = request.messages[-1]
            content = getattr(last_message, "content", None)
            user_command = content.strip() if isinstance(content, str) else ""

            if user_command.startswith("/"):
                agent = get_or_create_agent(conv_id)
                client = client_pool[conv_id]

                async for sse_event in handle_slash_command(
                    user_command,
                    conv_id,
                    client,
                    agent,
                    selected_model,
                    selected_temperature,
                    selected_max_tokens,
                    model_preferences,
                    temperature_preferences,
                    max_tokens_preferences,
                    token_usage,
                ):
                    yield sse_event
                # Exit after slash command - don't continue
            else:
                # Process normal query
                async for event in process_normal_query():
                    yield event
        else:
            # Process normal query
            async for event in process_normal_query():
                yield event

    async def process_normal_query() -> AsyncGenerator[dict[str, Any], None]:
        """Process normal query (non-slash-command)."""
        from openbb_ai import get_widget_data
        from openbb_ai.models import WidgetRequest
        from .conversation_manager import format_messages_for_llm
        from .tool_executor import execute_tool, get_tool_definitions
        from .streaming_handler import stream_llm_with_tools, generate_sse_events

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

        # CHECK IF WE NEED TO FETCH WIDGET DATA (EARLY EXIT)
        last_message = request.messages[-1] if request.messages else None

        selected_widget_stage_path = None  # Initialize widget context variables
        widget_context_str = ""
        widget_for_citations = None
        widget_input_args_for_citations = None
        widget_context_metadata: dict[str, Any] | None = None

        # Check if we have PRIMARY widgets explicitly added to context by the user
        # Secondary widgets are NOT used - only primary widgets are explicitly added
        has_primary_widgets = (
            request.widgets
            and request.widgets.primary
            and len(request.widgets.primary) > 0
        )
        if (
            last_message
            and last_message.role in ["human", "user"]
            and has_primary_widgets
        ):
            from .document_processor import DocumentProcessor

            # Emit reasoning step BEFORE document processing starts
            yield to_sse(
                reasoning_step(
                    "Reading document and metadata...",
                    event_type="INFO",
                )
            )

            client = client_pool[conv_id]
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
                widget_context_str = doc_result["widget_context_str"]
                widget_for_citations = doc_result["widget_for_citations"]
                widget_input_args_for_citations = doc_result[
                    "widget_input_args_for_citations"
                ]
                widget_context_metadata = doc_result["widget_context_metadata"]
                selected_widget_stage_path = doc_result["stage_path"]
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
                    # The widget data will come back in a subsequent request as a tool message
                    return
                # If no widget requests, fall through to normal message processing
                pass  # Continue to normal message processing below

        # Normal message processing - ALWAYS runs (after widget handling if any)
        # This handles both: messages with widgets AND simple messages without widgets
        try:
            _ = get_or_create_agent(conv_id)
            client = client_pool[conv_id]

            current_messages = []
            # Widget context variables now initialized at top of function scope

            # LOAD CONVERSATION HISTORY FROM CACHE FIRST
            # Retry once if session expired
            try:
                cached_messages = await run_in_thread(client.get_messages, conv_id)
            except Exception as e:
                if is_session_expired_error(e):
                    logger.warning("Session expired, refreshing client...")
                    client = refresh_client(conv_id)
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
                            {"is_tool_result": is_tool_result}
                            if is_tool_result
                            else None
                        ),
                    }
                )

            # Prime the deduplication cache with existing history
            seed_message_signatures(conv_id, all_messages)

            request_messages_to_add = []
            has_new_user_message = False
            needs_response = False
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
                        data_sources = (message_input_args or {}).get(
                            "data_sources", []
                        ) or []

                        # Get the target widget to determine processing strategy
                        target_widget = None
                        known_filename = None
                        if message_input_args and data_sources and all_widgets:
                            target_widget = find_widget_by_uuid(
                                data_sources[0].get("widget_uuid")
                            )

                        # Process based on widget type
                        from .widget_handler import WidgetHandler

                        widget_handler = await WidgetHandler.instance()

                        if target_widget and widget_handler.is_document_widget(
                            target_widget
                        ):
                            doc_proc = DocumentProcessor.instance()
                            known_filename = doc_proc.extract_filename_from_widget(
                                target_widget
                            )

                            parsed_data = widget_handler.parse_widget_data(
                                message.data, conv_id, client, known_filename
                            )
                            doc_proc.trigger_snowflake_upload_for_widget_pdf(
                                client, conv_id
                            )
                        else:
                            parsed_data = message.data
                            if target_widget:
                                widget_result = (
                                    await widget_handler.process_widget_response(
                                        target_widget, message.data, conv_id, client
                                    )
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
                            widget_data_request = (
                                data_sources[0] if data_sources else {}
                            )
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
                                widget_description = getattr(
                                    target_widget, "description", None
                                )
                                widget_type = getattr(
                                    target_widget, "type", None
                                ) or getattr(target_widget, "kind", None)

                            widget_input_args_dict = None
                            if isinstance(widget_data_request, dict):
                                widget_input_args_dict = dict(
                                    widget_data_request.get("input_args", {}) or {}
                                )
                                widget_input_args_dict.setdefault(
                                    "conversation_id", conv_id
                                )

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
                                    candidate_value = (
                                        widget_input_args_dict or {}
                                    ).get(candidate_key)
                                    if candidate_value:
                                        document_label = candidate_value
                                        break

                            stage_path = (
                                (widget_input_args_dict or {}).get("stage_path")
                                or (widget_data_request or {}).get("stage_path")
                                or selected_widget_stage_path
                            )
                            if widget_input_args_dict is not None and stage_path:
                                widget_input_args_dict.setdefault(
                                    "stage_path", stage_path
                                )

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
                                widget_display_name
                                or document_label
                                or target_uuid
                                or "widget"
                            )
                            if widget_input_args_dict is not None:
                                widget_input_args_dict.setdefault(
                                    "widget_label", widget_label
                                )
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
                                metadata_lines.append(
                                    f"Widget Name: {widget_display_name}"
                                )
                            if target_uuid:
                                metadata_lines.append(f"Widget UUID: {target_uuid}")
                            if document_label:
                                metadata_lines.append(
                                    f"Document/File: {document_label}"
                                )
                            if stage_path:
                                metadata_lines.append(f"Stage Path: {stage_path}")
                            if widget_type:
                                metadata_lines.append(f"Widget Type: {widget_type}")
                            if widget_description:
                                metadata_lines.append(
                                    f"Widget Description: {widget_description}"
                                )

                            data_label = (
                                widget_display_name or document_label or "Widget Data"
                            )
                            widget_context_str = (
                                "\n".join(metadata_lines)
                                + f"\n\n--- Widget Data: {data_label} ---\n{parsed_data}\n------\n"
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
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                logger.debug(
                                    "Last user message is resend - forcing fresh response"
                                )

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
                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    logger.debug("Last user message needs a response")

            if widget_for_citations:
                if widget_input_args_for_citations is None:
                    widget_input_args_for_citations = {"conversation_id": conv_id}
                else:
                    widget_input_args_for_citations.setdefault(
                        "conversation_id", conv_id
                    )

            # Only process if we have a new user message OR need to respond to existing one
            if (
                not has_new_user_message
                and not widget_context_str
                and not needs_response
            ):
                # Return the last assistant message if it exists
                for msg in reversed(all_messages):
                    if msg["role"] == "assistant":
                        yield to_sse(message_chunk(msg["content"]))
                        return

                # If no assistant message found, acknowledge the situation
                yield to_sse(
                    message_chunk(
                        "I'm ready to help. Please ask me a question or use /help to see available commands."
                    )
                )
                return

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

                        # Store ONLY new messages in cache
                        await run_in_thread(
                            client.add_message,
                            conv_id,
                            msg_id,
                            msg_dict["role"],
                            msg_dict["content"],
                        )
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
                if os.environ.get("SNOWFLAKE_DEBUG"):
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
                    enriched_user_message += "\n\n" + widget_context_str
                elif widget_context_metadata:
                    # Widget selected but no full context string built yet - inject explicit selection
                    doc_name = widget_context_metadata.get("document_label", "unknown")
                    stage = widget_context_metadata.get("stage_path", "")
                    enriched_user_message += f"\n\nWidget provided document: {doc_name}"
                    if stage:
                        enriched_user_message += f" ({stage})"
            elif widget_context_metadata:
                # No user message at all but widget is selected
                doc_name = widget_context_metadata.get("document_label", "unknown")
                stage = widget_context_metadata.get("stage_path", "")
                enriched_user_message = f"Widget provided document: {doc_name}"
                if stage:
                    enriched_user_message += f" ({stage})"
                enriched_user_message += "\n\nPlease analyze this document."

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

            # Validate that we have messages to send
            if not current_messages:
                logger.error("No messages to process for conversation %s", conv_id)
                yield to_sse(message_chunk("❌ No messages to process"))
            else:
                supports_tools = selected_model not in NON_TOOL_CALLING_MODELS
                tools = get_tool_definitions(client) if supports_tools else None

                # Validate message structure when using tools
                if supports_tools and tools and current_messages:
                    # Remove trailing assistant messages when using tools
                    while current_messages and current_messages[-1]["role"] in (
                        "assistant",
                        "ai",
                    ):
                        current_messages.pop()

                    # Also ensure we're not sending tool results as the last message
                    while current_messages and "[Tool Result" in current_messages[
                        -1
                    ].get("content", ""):
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
                        if (
                            not last_msg
                            or last_msg.get("content") != current_request_user_msg
                        ):
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

                TOOL_CAPABILITY_PHRASES = [
                    "what tools",
                    "which tools",
                    "tool do you have",
                    "tooling",
                    "capabilities",
                    "available tools",
                    "available functions",
                    "what functions",
                    "tool access",
                    "list your tools",
                    "what can you do",
                    "show your tools",
                ]

                if tool_overview and any(
                    phrase in last_user_msg_lower for phrase in TOOL_CAPABILITY_PHRASES
                ):
                    capability_response = (
                        "Here are the tools I have available right now:\n\n"
                        + tool_overview
                    )
                    yield to_sse(message_chunk(capability_response))

                    assistant_entry = {
                        "role": "assistant",
                        "content": capability_response,
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
                    return

                from .system_prompt import build_system_prompt
                from .document_processor import DocumentProcessor

                try:
                    available_docs = await run_in_thread(client.list_cortex_documents)
                except Exception:
                    available_docs = None

                # Get document structure if available for this conversation
                doc_proc = DocumentProcessor.instance()
                document_structure = doc_proc.format_document_structure_for_llm(conv_id)

                system_prompt = build_system_prompt(
                    total_messages=len(all_messages),
                    current_messages=len(current_messages),
                    user_schema=user_schema,
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
                                ai_messages_formatted_tuples.append(
                                    ("user", msg["content"])
                                )
                                break

                # Shared state for streaming
                stream_state = {
                    "full_text": "",
                    "tool_calls": [],
                    "usage": None,
                    "citation_count": 0,
                    "citation_summaries": [],
                    "fatal_error": None,
                }

                # Stream LLM response
                if not supports_tools:
                    # For non-tool-calling models
                    stream_state["full_text"] = ""
                    stream_state["tool_calls"] = []

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
                        logger.error(
                            "SSE stream timed out after %s seconds", sse_timeout
                        )
                        stream_state["fatal_error"] = (
                            f"Stream timeout after {sse_timeout}s"
                        )
                        yield to_sse(
                            message_chunk(f"❌ Stream timed out after {sse_timeout}s")
                        )

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
                        token_usage[conv_id]["prompt_tokens"] += usage.get(
                            "prompt_tokens", 0
                        )
                        token_usage[conv_id]["completion_tokens"] += usage.get(
                            "completion_tokens", 0
                        )
                        token_usage[conv_id]["total_tokens"] += usage.get(
                            "total_tokens", 0
                        )
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
                            logger.error(
                                "Non-tool path - ERROR storing token_usage: %s", e
                            )

                    if stream_state["full_text"].strip() and not stream_state.get(
                        "fatal_error"
                    ):
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
                else:
                    # Tool-calling flow
                    for iteration in range(MAX_TOOL_ITERATIONS):
                        if iteration == MAX_TOOL_ITERATIONS - 1:
                            yield to_sse(
                                reasoning_step(
                                    "Max tool iterations reached, aborting to prevent a loop.",
                                    event_type="ERROR",
                                )
                            )
                            yield to_sse(
                                message_chunk(
                                    "❌ I seem to be stuck in a loop. Please try rephrasing your request."
                                )
                            )
                            break

                        stream_state["full_text"] = ""
                        stream_state["tool_calls"] = []
                        stream_state["citation_count"] = 0
                        stream_state["citation_summaries"] = []
                        stream_state["fatal_error"] = None
                        buffered_text_chunks = (
                            []
                        )  # Buffer text until we know if there's a tool call
                        buffered_events = (
                            []
                        )  # Buffer both text and citations to preserve inline order

                        # First, yield that we're thinking about what to do
                        if iteration == 0:
                            yield to_sse(
                                reasoning_step(
                                    "Analyzing request and determining required tools...",
                                    event_type="INFO",
                                )
                            )

                        generator = stream_llm_with_tools(
                            client,
                            ai_messages_formatted_tuples,
                            selected_model,
                            selected_temperature,
                            selected_max_tokens,
                            tools=tools,
                            conv_id=conv_id,
                            widget=widget_for_citations,
                            widget_input_args=widget_input_args_for_citations,
                        )

                        # Consume the generator and handle events immediately
                        async for event in generator:
                            if not event:
                                continue

                            event_type, event_data = event

                            if event_type == "text":
                                if isinstance(event_data, str):
                                    # ALWAYS buffer text - we need to check if it's a tool call first
                                    buffered_text_chunks.append(event_data)
                                    buffered_events.append(("text", event_data))
                                    stream_state["full_text"] += event_data
                            elif event_type == "reasoning_complete":
                                # Emit complete think block as a single reasoning event
                                if isinstance(event_data, str) and event_data.strip():
                                    yield to_sse(
                                        reasoning_step(event_data, event_type="INFO")
                                    )
                            elif event_type == "citation":
                                # Buffer citation to maintain inline order with text
                                buffered_events.append(("citation", event_data))
                                stream_state["citation_count"] = (
                                    stream_state.get("citation_count", 0) + 1
                                )
                                summary_payload = dict(
                                    getattr(event_data, "extra_details", {}) or {}
                                )
                                summary_payload.setdefault(
                                    "citation_id",
                                    getattr(event_data, "citation_id", None),
                                )
                                stream_state.setdefault(
                                    "citation_summaries", []
                                ).append(summary_payload)
                            elif event_type == "tool_call":
                                stream_state["tool_calls"].append(event_data)
                            elif event_type == "complete":
                                if isinstance(event_data, dict):
                                    stream_state["tool_calls"] = event_data.get(
                                        "tool_calls", []
                                    )
                                    stream_state["usage"] = event_data.get(
                                        "usage", None
                                    )
                                    stream_state["full_text"] = event_data.get(
                                        "text", stream_state["full_text"]
                                    )
                                break
                            else:
                                # Yield any other events (like reasoning steps from handler)
                                message = (
                                    event_data
                                    if isinstance(event_data, str)
                                    else str(event_data)
                                )
                                yield to_sse(reasoning_step(message, event_type="INFO"))

                        # Update token usage after completion
                        usage = stream_state.get("usage")
                        if isinstance(usage, dict):
                            if conv_id not in token_usage:
                                token_usage[conv_id] = {
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0,
                                    "total_tokens": 0,
                                    "api_requests": 0,
                                }

                            token_usage[conv_id]["prompt_tokens"] += usage.get(
                                "prompt_tokens", 0
                            )
                            token_usage[conv_id]["completion_tokens"] += usage.get(
                                "completion_tokens", 0
                            )
                            token_usage[conv_id]["total_tokens"] += usage.get(
                                "total_tokens", 0
                            )
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
                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    logger.error(
                                        "Error storing token_usage for %s: %s",
                                        conv_id,
                                        e,
                                    )

                        # Check if there are tool calls to process
                        if not stream_state["tool_calls"]:
                            # Check if the full_text contains a tool call (JSON format)
                            # When tools aren't passed to API, LLM outputs tool calls as text
                            # The JSON may be preceded by explanatory text
                            full_text = stream_state["full_text"].strip()
                            if '"tool"' in full_text and "{" in full_text:
                                # Find the JSON object in the text
                                start_idx = full_text.find("{")
                                if start_idx != -1:
                                    # Count braces to find matching end
                                    brace_count = 0
                                    end_idx = start_idx
                                    for i, char in enumerate(
                                        full_text[start_idx:], start_idx
                                    ):
                                        if char == "{":
                                            brace_count += 1
                                        elif char == "}":
                                            brace_count -= 1
                                            if brace_count == 0:
                                                end_idx = i + 1
                                                break

                                    json_str = full_text[start_idx:end_idx]
                                    try:
                                        parsed = json.loads(json_str)
                                        tool_name = parsed.get("tool")
                                        args = parsed.get("arguments", {})
                                        if tool_name:
                                            stream_state["tool_calls"] = [
                                                ToolCall(
                                                    id=str(uuid.uuid4()),
                                                    tool_type="function",
                                                    function=FunctionCall(
                                                        name=tool_name,
                                                        arguments=(
                                                            json.dumps(args)
                                                            if isinstance(args, dict)
                                                            else str(args)
                                                        ),
                                                    ),
                                                )
                                            ]
                                            # Clear the buffered text - it contained a tool call
                                            buffered_text_chunks.clear()
                                    except json.JSONDecodeError:
                                        pass

                        # Fallback: detect when LLM says "I'll run/rerun X" but didn't emit tool JSON
                        # This handles cases where the model describes intent without calling
                        if not stream_state["tool_calls"]:
                            full_text_lower = stream_state["full_text"].lower()
                            # Patterns like "I'll rerun the OCR", "I'll run ocr_image", "let me try ocr again"
                            tool_intent_patterns = [
                                (
                                    r"(?:i'll|let me|i will|going to)\s+(?:re)?run\s+(?:the\s+)?ocr",
                                    "ocr_image",
                                ),
                                (
                                    r"(?:i'll|let me|i will|going to)\s+(?:re)?run\s+ocr_image",
                                    "ocr_image",
                                ),
                                (
                                    r"(?:i'll|let me|i will|going to)\s+try\s+(?:the\s+)?ocr\s+again",
                                    "ocr_image",
                                ),
                                (
                                    r"(?:i'll|let me|i will|going to)\s+extract.*(?:chart|image|graph)",
                                    "ocr_image",
                                ),
                            ]
                            import re as re_module

                            for pattern, inferred_tool in tool_intent_patterns:
                                if re_module.search(pattern, full_text_lower):
                                    # LLM said it would run a tool - look for args in conversation
                                    # Find the last ocr_image args from conversation history
                                    last_tool_args = None
                                    for msg in reversed(all_messages):
                                        content = msg.get("content", "")
                                        if (
                                            isinstance(content, str)
                                            and "ocr_image" in content.lower()
                                        ):
                                            # Try to find image_stage_path in context
                                            stage_match = re_module.search(
                                                r"@[\w.]+\.[\w.]+\.[\w_]+/[^\s\)]+\.(?:jpeg|jpg|png)",
                                                content,
                                                re_module.IGNORECASE,
                                            )
                                            if stage_match:
                                                last_tool_args = {
                                                    "image_stage_path": stage_match.group(
                                                        0
                                                    ),
                                                    "return_as_chart": True,
                                                }
                                                break

                                    if last_tool_args:
                                        logger.info(
                                            f"[auto-tool] Detected intent to run {inferred_tool}, injecting tool call"
                                        )
                                        stream_state["tool_calls"] = [
                                            ToolCall(
                                                id=str(uuid.uuid4()),
                                                tool_type="function",
                                                function=FunctionCall(
                                                    name=inferred_tool,
                                                    arguments=json.dumps(
                                                        last_tool_args
                                                    ),
                                                ),
                                            )
                                        ]
                                        buffered_text_chunks.clear()
                                        buffered_events.clear()
                                    break

                        if not stream_state["tool_calls"]:
                            # No tool calls - this is a final response
                            # Yield buffered events (text and citations) in order
                            if buffered_events:
                                for event_type, event_data in buffered_events:
                                    if event_type == "text":
                                        yield to_sse(message_chunk(event_data))
                                    elif event_type == "citation":
                                        yield to_sse(citations([event_data]))
                                buffered_events.clear()
                                buffered_text_chunks.clear()

                            if stream_state[
                                "full_text"
                            ].strip() and not stream_state.get("fatal_error"):
                                assistant_entry = {
                                    "role": "assistant",
                                    "content": stream_state["full_text"],
                                    "details": {
                                        "message_type": "assistant_final",
                                    },
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
                            break

                        # WE HAVE TOOL CALLS - Process them NOW with reasoning steps
                        has_tool_calls = len(stream_state["tool_calls"]) > 0

                        for tool_call in stream_state["tool_calls"]:
                            tool_name = tool_call.function.name
                            tool_args_str = tool_call.function.arguments
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
                                    tool_args = (
                                        json.loads(tool_args_str)
                                        if tool_args_str
                                        else {}
                                    )
                                except json.JSONDecodeError:
                                    tool_args = {}

                                current_tool_output_for_llm, raw_tool_data = (
                                    await doc_proc.handle_get_widget_data_tool_call(
                                        tool_args=tool_args,
                                        widgets_primary=(
                                            request.widgets.primary
                                            if request.widgets
                                            else None
                                        ),
                                        widgets_secondary=(
                                            request.widgets.secondary
                                            if request.widgets
                                            else None
                                        ),
                                        client=client,
                                        conversation_id=conv_id,
                                    )
                                )

                                # If we got a widget_data_request, yield it for the UI to fetch
                                if (
                                    isinstance(raw_tool_data, dict)
                                    and "widget_data_request" in raw_tool_data
                                ):
                                    yield raw_tool_data["widget_data_request"]
                                    # The actual data will come back in a subsequent request
                                    # Skip normal tool result processing for this case
                                    continue
                            else:
                                # This is a generator, so we need to iterate
                                current_tool_output_for_llm = ""
                                raw_tool_data = None
                                tool_output_generator = execute_tool(
                                    tool_call, client, conv_id
                                )
                                try:
                                    async for event in tool_output_generator:
                                        # The generator will yield reasoning steps first, then the final result
                                        if isinstance(event, tuple) and len(event) == 2:
                                            (
                                                current_tool_output_for_llm,
                                                raw_tool_data,
                                            ) = event
                                        elif (
                                            isinstance(event, dict)
                                            and event.get("event") == "reasoning_step"
                                        ):
                                            yield event
                                        elif (
                                            isinstance(event, dict) and "event" in event
                                        ):
                                            # SSE events like chart artifacts, tables, etc.
                                            # These come from to_sse() wrapped objects
                                            yield event
                                        elif hasattr(event, "event") and hasattr(
                                            event, "data"
                                        ):
                                            # MessageArtifactSSE objects (charts, tables) from openbb_ai.helpers
                                            yield to_sse(event)
                                        else:
                                            # Log unexpected event types for debugging
                                            logger.debug(
                                                f"Unhandled tool event type: {type(event)}"
                                            )
                                except (
                                    Exception
                                ) as tool_exc:  # pragma: no cover - defensive
                                    error_message = (
                                        f"Error executing tool {tool_name}: {tool_exc}"
                                    )
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
                                raw_tool_data = raw_tool_data or {
                                    "error": "empty_result"
                                }

                            # Yield completion reasoning step
                            if "Error getting" in str(
                                current_tool_output_for_llm
                            ) or "Error:" in str(current_tool_output_for_llm):
                                yield to_sse(
                                    reasoning_step(
                                        f"Tool {tool_name} encountered an error.",
                                        event_type="ERROR",
                                    )
                                )
                            else:
                                if (
                                    tool_name
                                    in ["execute_query", "get_table_sample_data"]
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
                            tool_result_formatted = f"The result from {tool_name} is:\n{current_tool_output_for_llm}"

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

                        # Continue to get the final response
                        if has_tool_calls:
                            yield to_sse(
                                reasoning_step(
                                    "Processing results and generating response...",
                                    event_type="INFO",
                                )
                            )

                            # Continue, LLM will now interpret the tool results.
                            continue
                        else:
                            # No more tool calls - we're done
                            break

        except Exception as e:
            tb = traceback.format_exc()
            yield to_sse(reasoning_step(f"Error: {str(e)}\n{tb}", event_type="ERROR"))
            yield to_sse(message_chunk(f"❌ An error occurred: {str(e)}"))

        except Exception as e:
            tb = traceback.format_exc()
            yield to_sse(
                reasoning_step(
                    f"Error in message processing: {str(e)}\n{tb}", event_type="ERROR"
                )
            )
            yield to_sse(message_chunk(f"❌ An error occurred: {str(e)}"))

    async def sse_generator():
        async for event in execution_loop():
            yield event

    return EventSourceResponse(
        content=sse_generator(),
        media_type="text/event-stream",
    )
