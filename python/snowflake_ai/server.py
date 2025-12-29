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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb_ai import (
    QueryRequest,
)
from sse_starlette import EventSourceResponse

# These are the Rust bindings
from ._snowflake_ai import (
    SnowflakeAgent,
    SnowflakeAI,
)
from .logger import get_logger
from .slash_commands import handle_slash_command
from .helpers import (
    seed_message_signatures,
    should_store_message,
    to_sse,
    run_in_thread,
)
from .widgets import router as widgets_router

logger = get_logger(__name__)

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
# Cache last query results per conversation for charting
last_query_results: dict[str, list] = {}
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

        # Initialize/load conversation from AGENTS_CONVERSATIONS table
        default_settings = {
            "model": "openai-gpt-5-chat",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
        try:
            result = client.get_or_create_conversation(
                conversation_id, json.dumps(default_settings)
            )
            # If conversation existed, load its settings
            if result:
                try:
                    existing_settings = json.loads(result)
                    logger.debug(
                        "Loaded conversation settings for %s: %s",
                        conversation_id,
                        existing_settings,
                    )
                    if "model" in existing_settings:
                        model_preferences[conversation_id] = existing_settings["model"]
                    if "temperature" in existing_settings:
                        temperature_preferences[conversation_id] = float(
                            existing_settings["temperature"]
                        )
                    if "max_tokens" in existing_settings:
                        max_tokens_preferences[conversation_id] = int(
                            existing_settings["max_tokens"]
                        )
                    if "token_usage" in existing_settings:
                        token_usage[conversation_id] = existing_settings["token_usage"]
                    # Restore database and schema context
                    db = existing_settings.get("database")
                    sch = existing_settings.get("schema")
                    if db or sch:
                        logger.debug(
                            "Restoring context for conversation %s: database=%s, schema=%s",
                            conversation_id,
                            db,
                            sch,
                        )
                        client.use_conversation_context(db, sch)
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.debug(
                        "Error parsing conversation settings for %s: %s",
                        conversation_id,
                        e,
                    )
        except Exception as e:
            logger.debug("Error initializing conversation %s: %s", conversation_id, e)

        # Load preferences from AGENTS_CONTEXT_OBJECTS if not already loaded
        try:
            if conversation_id not in model_preferences:
                cached_model = client.get_conversation_data(
                    conversation_id, "model_preference"
                )
                if cached_model:
                    model_preferences[conversation_id] = cached_model

            if conversation_id not in temperature_preferences:
                cached_temperature = client.get_conversation_data(
                    conversation_id, "temperature_preference"
                )
                if cached_temperature:
                    temperature_preferences[conversation_id] = float(cached_temperature)

            if conversation_id not in max_tokens_preferences:
                cached_max_tokens = client.get_conversation_data(
                    conversation_id, "max_tokens_preference"
                )
                if cached_max_tokens:
                    max_tokens_preferences[conversation_id] = int(cached_max_tokens)

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
    import time

    request_start_time = time.time()
    logger.debug("[REQUEST START] New /query request at %s", request_start_time)

    if request.messages:
        last_msg = request.messages[-1]
        content = (
            getattr(last_msg, "content", "")[:100]
            if hasattr(last_msg, "content")
            else ""
        )
        logger.debug("[REQUEST START] Last message: %s...", content)

    # Extract conversation ID IMMEDIATELY when request comes in
    conv_id = request_obj.headers.get("x-trace-id") or "default"

    logger.debug("[REQUEST START] Conversation ID: %s", conv_id)

    # Ensure agent and preferences are loaded from cache BEFORE reading them
    get_or_create_agent(conv_id)

    # Get all preferences IMMEDIATELY after we have conv_id
    selected_model = model_preferences.get(conv_id, "openai-gpt-5-chat")
    selected_temperature = temperature_preferences.get(conv_id, 0.7)
    selected_max_tokens = max_tokens_preferences.get(conv_id, 4096)

    async def execution_loop():
        """Main execution loop."""
        exec_id = str(uuid.uuid4())[:8]

        logger.debug(
            "[EXEC %s] execution_loop started for conv_id=%s", exec_id, conv_id
        )

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
                    if (
                        isinstance(event, dict)
                        and event.get("event") == "copilotMessageArtifact"
                    ):
                        data_str = str(event.get("data", ""))[:400]
                        logger.debug(
                            "[EXEC %s] YIELDING ARTIFACT from process_normal_query: %s",
                            exec_id,
                            data_str,
                        )
                    yield event
        else:
            # Process normal query
            async for event in process_normal_query():
                if (
                    isinstance(event, dict)
                    and event.get("event") == "copilotMessageArtifact"
                ):
                    data_str = str(event.get("data", ""))[:400]
                    logger.debug(
                        "[EXEC %s] YIELDING ARTIFACT from process_normal_query: %s",
                        exec_id,
                        data_str,
                    )
                yield event

    async def process_normal_query() -> AsyncGenerator[dict[str, Any], None]:
        """Process normal query (non-slash-command)."""
        from .query_processing import (
            handle_primary_widgets,
            load_conversation_history,
            process_incoming_messages,
            prepare_llm_context,
            stream_response_no_tools,
            execute_single_tool,
        )
        from .helpers import format_tool_overview
        from .tool_executor import get_tool_definitions
        from openbb_ai import reasoning_step
        from .streaming_handler import message_chunk, stream_llm_with_tools, citations
        from ._snowflake_ai import ToolCall, FunctionCall

        # 1. Handle Primary Widgets
        client = client_pool[conv_id]

        widget_context_result = None
        async for event in handle_primary_widgets(request, conv_id, client):
            if "should_return" in event:
                widget_context_result = event
            else:
                yield event

        if widget_context_result and widget_context_result.get("should_return"):
            return

        # Extract context variables
        widget_context_str = (
            widget_context_result.get("widget_context_str", "")
            if widget_context_result
            else ""
        )
        widget_for_citations = (
            widget_context_result.get("widget_for_citations")
            if widget_context_result
            else None
        )
        widget_input_args_for_citations = (
            widget_context_result.get("widget_input_args_for_citations")
            if widget_context_result
            else None
        )
        widget_context_metadata = (
            widget_context_result.get("widget_context_metadata")
            if widget_context_result
            else None
        )
        selected_widget_stage_path = (
            widget_context_result.get("selected_widget_stage_path")
            if widget_context_result
            else None
        )

        # 2. Normal Message Processing
        try:
            _ = get_or_create_agent(conv_id)
            # client is already retrieved above

            # 3. Load Conversation History
            all_messages = await load_conversation_history(
                client, conv_id, refresh_client
            )

            # 4. Process Incoming Messages
            (
                all_messages,
                has_new_user_message,
                needs_response,
                widget_for_citations,
                widget_input_args_for_citations,
                widget_context_str,
                widget_context_metadata,
            ) = await process_incoming_messages(
                request,
                all_messages,
                conv_id,
                client,
                selected_widget_stage_path,
                existing_widget_context_str=widget_context_str,
                existing_widget_context_metadata=widget_context_metadata,
                existing_widget_for_citations=widget_for_citations,
                existing_widget_input_args=widget_input_args_for_citations,
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

            # 5. Prepare LLM Context
            supports_tools = selected_model not in NON_TOOL_CALLING_MODELS
            tools = get_tool_definitions(client) if supports_tools else None

            # Check for tool capability query
            last_user_msg_lower = ""
            for msg in reversed(request.messages):
                if msg.role in ["human", "user"]:
                    content = getattr(msg, "content", None)
                    if content:
                        last_user_msg_lower = (
                            content if isinstance(content, str) else str(content)
                        ).lower()
                    break

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
                    "Here are the tools I have available right now:\n\n" + tool_overview
                )
                yield to_sse(message_chunk(capability_response))

                # Store response
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

            ai_messages_formatted_tuples, tool_overview = await prepare_llm_context(
                request,
                all_messages,
                widget_context_str,
                widget_context_metadata,
                conv_id,
                client,
                selected_model,
                supports_tools,
                tools,
            )

            # Shared state for streaming
            stream_state = {
                "full_text": "",
                "tool_calls": [],
                "usage": None,
                "citation_count": 0,
                "citation_summaries": [],
                "fatal_error": None,
            }

            # 6. Stream Response
            if not supports_tools:
                async for event in stream_response_no_tools(
                    client,
                    ai_messages_formatted_tuples,
                    selected_model,
                    selected_temperature,
                    selected_max_tokens,
                    conv_id,
                    widget_for_citations,
                    widget_input_args_for_citations,
                    token_usage,
                    all_messages,
                ):
                    yield event
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
                    buffered_text_chunks = []
                    buffered_events = []

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

                    # Track if response looks like a JSON tool call (suppress streaming in that case)
                    looks_like_tool_call = False

                    async for event in generator:
                        if not event:
                            continue
                        event_type, event_data = event

                        if event_type == "text":
                            if isinstance(event_data, str):
                                buffered_text_chunks.append(event_data)
                                buffered_events.append(("text", event_data))
                                stream_state["full_text"] += event_data

                                # Check early if this looks like a JSON tool call
                                # If so, don't stream - we'll handle it after loop
                                current_text = stream_state["full_text"].strip()
                                if not looks_like_tool_call:
                                    # Check for JSON object start or tool pattern
                                    if (
                                        current_text.startswith("{")
                                        or '"tool"' in current_text
                                    ):
                                        looks_like_tool_call = True
                                        logger.debug(
                                            "Detected potential tool call JSON, suppressing stream"
                                        )

                                # Only stream if NOT a tool call
                                if not looks_like_tool_call:
                                    # Stream immediately for real-time rendering
                                    yield to_sse(message_chunk(event_data))
                        elif event_type == "sql":
                            if isinstance(event_data, str):
                                sql_block = event_data.strip()
                                if sql_block:
                                    formatted_sql = f"```sql\n{sql_block}\n```"
                                    buffered_text_chunks.append(formatted_sql)
                                    buffered_events.append(("text", formatted_sql))
                                    stream_state["full_text"] += formatted_sql
                                    if not looks_like_tool_call:
                                        yield to_sse(message_chunk(formatted_sql))
                        elif event_type == "reasoning_complete":
                            if isinstance(event_data, str) and event_data.strip():
                                yield to_sse(
                                    reasoning_step(event_data, event_type="INFO")
                                )
                        elif event_type == "citation":
                            buffered_events.append(("citation", event_data))
                            stream_state["citation_count"] = (
                                stream_state.get("citation_count", 0) + 1
                            )
                            summary_payload = dict(
                                getattr(event_data, "extra_details", {}) or {}
                            )
                            summary_payload.setdefault(
                                "citation_id", getattr(event_data, "citation_id", None)
                            )
                            stream_state.setdefault("citation_summaries", []).append(
                                summary_payload
                            )
                            # Only stream citation if NOT a tool call
                            if not looks_like_tool_call:
                                yield to_sse(citations([event_data]))
                        elif event_type == "tool_call":
                            stream_state["tool_calls"].append(event_data)
                        elif event_type == "complete":
                            if isinstance(event_data, dict):
                                stream_state["tool_calls"] = event_data.get(
                                    "tool_calls", []
                                )
                                stream_state["usage"] = event_data.get("usage", None)
                                stream_state["full_text"] = event_data.get(
                                    "text", stream_state["full_text"]
                                )
                            break
                        else:
                            message = (
                                event_data
                                if isinstance(event_data, str)
                                else str(event_data)
                            )
                            yield to_sse(reasoning_step(message, event_type="INFO"))

                    # Update token usage
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
                        try:
                            await run_in_thread(
                                client.set_conversation_data,
                                conv_id,
                                "token_usage",
                                json.dumps(token_usage[conv_id]),
                            )
                        except Exception as e:
                            logger.error(
                                "Error storing token_usage for %s: %s", conv_id, e
                            )

                    # Check for tool calls in text if not explicit
                    if not stream_state["tool_calls"]:
                        full_text = stream_state["full_text"].strip()
                        logger.info(
                            "Checking for JSON tool call. Has '\"tool\"': %s, text len: %d, preview: %s",
                            '"tool"' in full_text,
                            len(full_text),
                            full_text[:200] if full_text else "(empty)",
                        )
                        # Look for JSON tool call pattern: {"tool": "...", "arguments": {...}}
                        if '"tool"' in full_text:
                            logger.info("Found '\"tool\"' in text, attempting to parse")

                            # Try to fix common LLM JSON mistakes before parsing
                            fixed_text = full_text
                            # Fix missing [ before array values like "key": 1, 2]
                            import re as fix_re

                            # Pattern: "key": value, value] -> "key": [value, value]
                            fixed_text = fix_re.sub(
                                r'("page_numbers"\s*:\s*)(\d+(?:\s*,\s*\d+)*)\]',
                                r"\1[\2]",
                                fixed_text,
                            )
                            # Also fix other array-like patterns
                            fixed_text = fix_re.sub(
                                r'("[\w_]+"\s*:\s*)(\d+(?:\s*,\s*\d+)+)\]',
                                r"\1[\2]",
                                fixed_text,
                            )

                            if fixed_text != full_text:
                                logger.info(
                                    "Fixed malformed JSON: %s", fixed_text[:200]
                                )

                            # Try to parse the (possibly fixed) text as JSON
                            try:
                                parsed = json.loads(fixed_text)
                                tool_name = parsed.get("tool")
                                args = parsed.get("arguments", {})
                                if tool_name:
                                    logger.info(
                                        "JSON parse succeeded: tool=%s", tool_name
                                    )
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
                                    buffered_text_chunks.clear()
                                    buffered_events.clear()
                            except json.JSONDecodeError as je:
                                logger.warning(
                                    "JSON parse failed even after fixes: %s", je
                                )
                                # Try to extract JSON from text with brace matching
                                tool_pos = fixed_text.find('"tool"')
                                if tool_pos > 0:
                                    start = fixed_text.rfind("{", 0, tool_pos)
                                    if start != -1:
                                        depth = 0
                                        end = start
                                        for i in range(start, len(fixed_text)):
                                            if fixed_text[i] == "{":
                                                depth += 1
                                            elif fixed_text[i] == "}":
                                                depth -= 1
                                                if depth == 0:
                                                    end = i + 1
                                                    break
                                        json_str = fixed_text[start:end]
                                        logger.info(
                                            "Trying brace-matched JSON: %s",
                                            json_str[:200],
                                        )
                                        try:
                                            parsed = json.loads(json_str)
                                            tool_name = parsed.get("tool")
                                            args = parsed.get("arguments", {})
                                            if tool_name:
                                                logger.info(
                                                    "Brace-matched parse succeeded: tool=%s",
                                                    tool_name,
                                                )
                                                stream_state["tool_calls"] = [
                                                    ToolCall(
                                                        id=str(uuid.uuid4()),
                                                        tool_type="function",
                                                        function=FunctionCall(
                                                            name=tool_name,
                                                            arguments=(
                                                                json.dumps(args)
                                                                if isinstance(
                                                                    args, dict
                                                                )
                                                                else str(args)
                                                            ),
                                                        ),
                                                    )
                                                ]
                                                buffered_text_chunks.clear()
                                                buffered_events.clear()
                                        except json.JSONDecodeError as je2:
                                            logger.warning(
                                                "Brace-matched parse also failed: %s",
                                                je2,
                                            )

                    # Fallback intent detection
                    if not stream_state["tool_calls"]:
                        full_text_lower = stream_state["full_text"].lower()
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
                                last_tool_args = None
                                for msg in reversed(all_messages):
                                    content = msg.get("content", "")
                                    if (
                                        isinstance(content, str)
                                        and "ocr_image" in content.lower()
                                    ):
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
                                    logger.debug(
                                        "[auto-tool] Detected intent to run %s, injecting tool call",
                                        inferred_tool,
                                    )
                                    stream_state["tool_calls"] = [
                                        ToolCall(
                                            id=str(uuid.uuid4()),
                                            tool_type="function",
                                            function=FunctionCall(
                                                name=inferred_tool,
                                                arguments=json.dumps(last_tool_args),
                                            ),
                                        )
                                    ]
                                    buffered_text_chunks.clear()
                                    buffered_events.clear()
                                break

                    # Detect if LLM output planning/reasoning instead of tool call
                    # This is REASONING, not final output - emit as reasoning_step and continue
                    if not stream_state["tool_calls"]:
                        full_text = stream_state["full_text"].strip()
                        planning_patterns = [
                            r"(?:let'?s|i'?ll|i will|i need to|going to)\s+(?:inspect|check|look at|examine|retrieve|fetch|get|query|run|use|call)",
                            r"(?:let me|allow me to)\s+(?:inspect|check|look|examine|retrieve|fetch|get|query|run)",
                            r"to (?:proceed|continue|answer|find|get|query).*(?:i need|we need|let's|i'll)",
                            r"(?:first|next),?\s+(?:i'?ll|let'?s|i need to|we need to)",
                            r"we'?ll use (?:these|those|the)",
                            r"once i have (?:those|these|the)",
                        ]
                        is_planning_text = False
                        import re as re_mod

                        for pattern in planning_patterns:
                            if re_mod.search(pattern, full_text.lower()):
                                is_planning_text = True
                                break

                        if is_planning_text and iteration < MAX_TOOL_ITERATIONS - 2:
                            # This is REASONING - emit as reasoning event, not final text
                            yield to_sse(reasoning_step(full_text, event_type="INFO"))

                            # Add to messages and nudge LLM to call the tool
                            ai_messages_formatted_tuples.append(
                                ("assistant", full_text)
                            )
                            ai_messages_formatted_tuples.append(
                                (
                                    "user",
                                    'Now call the tool. Output ONLY: {"tool": "...", "arguments": {...}}',
                                )
                            )
                            buffered_text_chunks.clear()
                            buffered_events.clear()
                            continue  # Re-run to get the actual tool call

                    # Log tool call state before final decision
                    logger.debug(
                        "Tool call state: has_tool_calls=%s, full_text_preview=%s",
                        bool(stream_state["tool_calls"]),
                        (
                            stream_state["full_text"][:100]
                            if stream_state["full_text"]
                            else "(empty)"
                        ),
                    )

                    if not stream_state["tool_calls"]:
                        # Final response - no tool calls detected
                        logger.info(
                            "No tool calls detected after all checks, outputting as final response. Text preview: %s",
                            (
                                stream_state["full_text"][:100]
                                if stream_state["full_text"]
                                else "(empty)"
                            ),
                        )
                        # Only yield buffered events if we suppressed streaming (thought it was a tool call but wasn't)
                        if buffered_events and looks_like_tool_call:
                            logger.info(
                                "Yielding %d buffered events as final text (was suppressed as potential tool call)",
                                len(buffered_events),
                            )
                            for event_type, event_data in buffered_events:
                                if event_type == "text":
                                    yield to_sse(message_chunk(event_data))
                                elif event_type == "citation":
                                    yield to_sse(citations([event_data]))
                            buffered_events.clear()
                            buffered_text_chunks.clear()

                        if stream_state["full_text"].strip() and not stream_state.get(
                            "fatal_error"
                        ):
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
                        break

                    # Process Tool Calls
                    has_tool_calls = len(stream_state["tool_calls"]) > 0
                    tool_names_in_batch = [
                        tc.function.name for tc in stream_state["tool_calls"]
                    ]
                    text2sql_will_auto_execute = "text2sql" in tool_names_in_batch

                    logger.info(
                        "Executing %d tool calls: %s",
                        len(stream_state["tool_calls"]),
                        tool_names_in_batch,
                    )

                    for tool_call in stream_state["tool_calls"]:
                        logger.info(
                            "Starting execution of tool: %s with args: %s",
                            tool_call.function.name,
                            (
                                tool_call.function.arguments[:200]
                                if tool_call.function.arguments
                                else "(none)"
                            ),
                        )
                        async for event in execute_single_tool(
                            tool_call,
                            client,
                            conv_id,
                            request,
                            all_messages,
                            ai_messages_formatted_tuples,
                            text2sql_will_auto_execute,
                        ):
                            yield event
                        logger.info(
                            "Finished execution of tool: %s", tool_call.function.name
                        )

                    if has_tool_calls:
                        yield to_sse(
                            reasoning_step(
                                "Processing results and generating response...",
                                event_type="INFO",
                            )
                        )
                        continue
                    else:
                        break

        except Exception as e:
            tb = traceback.format_exc()
            yield to_sse(reasoning_step(f"Error: {str(e)}\n{tb}", event_type="ERROR"))
            yield to_sse(message_chunk(f"❌ An error occurred: {str(e)}"))

    async def sse_generator():
        """SSE generator with logging and cancellation handling."""
        request_id = str(uuid.uuid4())[:8]
        event_counter = 0
        artifact_count = 0
        logger.debug(
            "[REQUEST %s] SSE generator started for conv_id=%s",
            request_id,
            conv_id,
        )

        try:
            async for event in execution_loop():
                # Check if client disconnected
                if await request_obj.is_disconnected():
                    logger.info(
                        "[REQUEST %s] Client disconnected, stopping stream", request_id
                    )
                    break

                event_counter += 1
                # FINAL SSE OUTPUT: Log every event for tracing
                event_type = (
                    event.get("event", "unknown")
                    if isinstance(event, dict)
                    else "non-dict"
                )
                logger.debug(
                    "[REQUEST %s] [SSE EVENT #%s] Type: %s",
                    request_id,
                    event_counter,
                    event_type,
                )

                if (
                    isinstance(event, dict)
                    and event.get("event") == "copilotMessageArtifact"
                ):
                    artifact_count += 1
                    data_str = str(event.get("data", ""))[:500]
                    logger.debug(
                        "[REQUEST %s] [SSE WIRE ARTIFACT #%s] Sending: %s",
                        request_id,
                        artifact_count,
                        data_str,
                    )
                yield event
        except asyncio.CancelledError:
            logger.info("[REQUEST %s] Request cancelled by client", request_id)
            return
        except Exception as e:
            logger.error("[REQUEST %s] SSE generator error: %s", request_id, e)

        logger.debug(
            "[REQUEST %s] [SSE COMPLETE] Total events: %s, artifacts: %s",
            request_id,
            event_counter,
            artifact_count,
        )

    return EventSourceResponse(
        content=sse_generator(),
        media_type="text/event-stream",
    )
