"""OpenBB AI Agent server for Snowflake AI_COMPLETE."""

# flake8: noqa: PLR0911, PLR0912, T201
# pylint: disable = R0911, R0912, R0914, R0915, R0917, C0103, C0415, E0611, W0718

import asyncio
import json
import os
import re
import uuid
import threading
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb_ai import (
    QueryRequest,
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
from .helpers import to_sse
from .models import SchemaItem
from .widgets import router as widgets_router

AGENT_BASE_URL = os.environ.get("AGENT_BASE_URL", "http://0.0.0.0:6975")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(widgets_router)


def cleanup_text(text: str) -> str:
    """Remove extra backslashes from text."""
    # This regex finds and replaces escaped underscores and backticks
    # that are not part of a larger escaped sequence.
    return re.sub(r"\\([_`])", r"\1", text)


# Connection pool indexed by conversation_id
agent_pool: dict[str, SnowflakeAgent] = {}
# Store model preferences per conversation
model_preferences: dict[str, str] = {}
# Store temperature preferences per conversation
temperature_preferences: dict[str, float] = {}
# Store max_tokens preferences per conversation
max_tokens_preferences: dict[str, int] = {}
# Snowflake SQL Cache Client pool.
client_pool: dict[str, SnowflakeAI] = {}
# Track token usage per conversation
token_usage: dict[str, dict[str, int]] = {}
MAX_TOOL_ITERATIONS = 10

NON_TOOL_CALLING_MODELS = {
    "llama4-maverick",
    "llama3.1-8b",
    "llama3.1-70b",
    "llama3.1-405b",
    "deepseek-r1",
    "mistral-7b",
    "mistral-large",
    "mistral-large2",
    "snowflake-llama-3.3-70b",
    # Add more models that don't support tool calling
}


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
                        import json

                        token_usage[conversation_id] = json.loads(cached_usage)
                except Exception:
                    pass

        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(
                    f"[DEBUG] Error loading preferences from cache for {conversation_id}: {e}"
                )

    return agent_pool[conversation_id]


async def run_in_thread(func, *args, **kwargs):
    """Wrap a blocking function to run in a background thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


def is_iterator(value: object) -> bool:
    """Check if a value is an iterator."""
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


async def iterate_sync_generator(generator: Iterator):
    """Consume a sync generator in a background thread and yield results."""
    loop = asyncio.get_event_loop()
    queue = asyncio.Queue()

    def consume_generator():
        try:
            for chunk in generator:
                if chunk is not None:
                    # Yield the raw chunk object instead of str(chunk)
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


async def shutdown_event():
    """Close all active client connections on shutdown."""
    for client in client_pool.values():
        try:
            await run_in_thread(client.close)
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print("[DEBUG] Closed a SnowflakeAI client connection.")
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Error closing a client connection: {e}")


app.add_event_handler(event_type="shutdown", func=shutdown_event)


@app.get("/table_info/{table_name}", response_model=list[SchemaItem])
async def get_table_info(table_name: str, request: Request):
    """Get table info for a specific table."""
    conversation_id = request.headers.get("x-trace-id") or "default"
    get_or_create_agent(conversation_id)  # Ensures client is created and cached
    client = client_pool[conversation_id]
    try:
        info_str = await run_in_thread(client.get_table_info, table_name)
        return json.loads(info_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
    print(widgets_path)
    with open(widgets_path, encoding="utf-8") as file:
        widgets = json.load(file)
    return widgets


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

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print("[DEBUG] ========== CHAT CONFIGURATION ==========")
        print(f"[DEBUG] Conversation ID: {conv_id}")
        print(f"[DEBUG] Selected Model: {selected_model}")
        print(f"[DEBUG] Selected Temperature: {selected_temperature}")
        print(f"[DEBUG] Selected Max Tokens: {selected_max_tokens}")
        print(f"[DEBUG] Model Preferences: {model_preferences}")
        print(f"[DEBUG] Temperature Preferences: {temperature_preferences}")
        print(f"[DEBUG] Max Tokens Preferences: {max_tokens_preferences}")
        total_conversations = len(
            set(
                list(model_preferences.keys())
                + list(temperature_preferences.keys())
                + list(max_tokens_preferences.keys())
            )
        )
        print(f"[DEBUG] Total Conversations: {total_conversations}")
        print("[DEBUG] ==========================================")

    async def execution_loop():
        """Main execution loop."""
        from .slash_commands import handle_slash_command

        # Check for slash commands FIRST
        if request.messages and request.messages[-1].role == "human":
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

    async def process_normal_query():
        """Process normal query (non-slash-command)."""
        from openbb_ai import get_widget_data
        from openbb_ai.models import WidgetRequest
        from .helpers import parse_widget_data
        from .conversation_manager import format_messages_for_llm
        from .tool_executor import execute_tool, get_tool_definitions
        from .streaming_handler import stream_llm_with_tools, generate_sse_events

        # CHECK IF WE NEED TO FETCH WIDGET DATA (EARLY EXIT)
        last_message = request.messages[-1] if request.messages else None
        if (
            last_message
            and last_message.role == "human"
            and request.widgets
            and request.widgets.primary
        ):
            widget_requests = []
            for widget in request.widgets.primary:
                widget_req = WidgetRequest(
                    widget=widget,
                    input_arguments=(
                        {param.name: param.current_value for param in widget.params}
                        if hasattr(widget, "params")
                        else {}
                    ),
                )
                widget_requests.append(widget_req)

            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Early exit to fetch {len(widget_requests)} widgets")

            # Yield widget data
            result = get_widget_data(widget_requests)
            yield result.model_dump()
            # Exit by not continuing to else block

        else:
            # Normal message processing (happens AFTER widget data is fetched)
            try:
                _ = get_or_create_agent(conv_id)
                client = client_pool[conv_id]

                current_messages = []
                widget_context_str = ""
                widget_for_citations = None
                widget_input_args_for_citations = None

                # LOAD CONVERSATION HISTORY FROM CACHE FIRST
                cached_messages = await run_in_thread(client.get_messages, conv_id)

                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Loaded {len(cached_messages)} messages from cache")

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

                request_messages_to_add = []
                has_new_user_message = False
                needs_response = False

                for idx, message in enumerate(request.messages):
                    # Handle tool messages (widget data comes back as tool messages)
                    if message.role == "tool":
                        # Only process if it has data
                        if hasattr(message, "data") and message.data:
                            # Always parse to ensure pdf_text_blocks is populated
                            parsed_data = parse_widget_data(message.data, conv_id)

                            # Only add context if it's the last message
                            if idx == len(request.messages) - 1:
                                widget_context_str = f"Use the following data to answer the question:\n\n--- Data ---\n{parsed_data}\n------\n"

                                # Extract widget info for citations
                                if (
                                    hasattr(message, "input_arguments")
                                    and message.input_arguments
                                ):
                                    data_sources = message.input_arguments.get(
                                        "data_sources", []
                                    )
                                    if data_sources:
                                        widget_data_request = data_sources[0]
                                        target_uuid = widget_data_request.get(
                                            "widget_uuid"
                                        )

                                        if (
                                            target_uuid
                                            and request.widgets
                                            and request.widgets.primary
                                        ):
                                            for w in request.widgets.primary:
                                                if str(w.uuid) == target_uuid:
                                                    widget_for_citations = w
                                                    widget_input_args_for_citations = (
                                                        widget_data_request.get(
                                                            "input_args", {}
                                                        )
                                                    )
                                                    break

                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(
                                        f"[DEBUG] Widget for citations found: {widget_for_citations is not None}"
                                    )
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
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            f"[DEBUG] Skipping duplicate message: {message.role} - {message_content[:50]}..."
                                        )
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
                                    print(
                                        "[DEBUG] Last user message is a resend/explicit send - forcing a fresh response"
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
                                            and "[Tool Result"
                                            not in cached_msg["content"]
                                        ):
                                            # Another user message without tool result means no response to previous
                                            break

                                if not has_response:
                                    needs_response = True
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            "[DEBUG] Last user message needs a response"
                                        )

                # Only process if we have a new user message OR need to respond to existing one
                if (
                    not has_new_user_message
                    and not widget_context_str
                    and not needs_response
                ):
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            "[DEBUG] No new user message to process and last message already has response"
                        )

                    # Return the last assistant message if it exists
                    for msg in reversed(all_messages):
                        if msg["role"] == "assistant":
                            yield to_sse(message_chunk(msg["content"]))
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                print(
                                    f"[DEBUG] Returning cached assistant response: {msg['content'][:100]}..."
                                )
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
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Adding {len(request_messages_to_add)} new messages to cache"
                        )

                    for message in request_messages_to_add:
                        msg_id = str(uuid.uuid4())
                        msg_dict = {
                            "role": message.role,
                            "content": (
                                message.content
                                if isinstance(message.content, str)
                                else str(message.content)
                            ),
                            "details": None,
                        }
                        all_messages.append(msg_dict)

                        # Store ONLY new messages in cache
                        await run_in_thread(
                            client.add_message,
                            conv_id,
                            msg_id,
                            msg_dict["role"],
                            msg_dict["content"],
                        )
                else:
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] No new messages to add - all {len(request.messages)} messages already in cache"
                        )

                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Total messages in history: {len(all_messages)}")

                # Keep a sliding window of recent messages, but the full history is available if needed
                current_messages = []

                # Determine if we need full history based on the user's query
                last_user_msg = ""
                for msg in reversed(all_messages):
                    if msg["role"] in ["human", "user"]:
                        last_user_msg = msg["content"].lower()
                        break

                # Check if user is asking about conversation history or previous data
                # Expanded keywords to catch more cases where full history is needed
                needs_full_history = any(
                    phrase in last_user_msg
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
                    MAX_LLM_CONTEXT_MESSAGES = (
                        200  # Increased to ensure we get everything
                    )
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] User query references history/context - including up to {MAX_LLM_CONTEXT_MESSAGES} messages"
                        )

                    # Include all messages for history queries
                    current_messages = all_messages[-MAX_LLM_CONTEXT_MESSAGES:]

                    # Log what tool results are in the context
                    tool_results_count = sum(
                        1
                        for msg in current_messages
                        if msg.get("role") == "user"
                        and "[Tool Result" in msg.get("content", "")
                    )
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Including {tool_results_count} tool results in context"
                        )
                else:
                    # Normal query - use sliding window for efficiency
                    SLIDING_WINDOW_SIZE = 30
                    all_tool_results = []
                    for msg in all_messages:
                        if msg["role"] in ["user"] and "[Tool Result" in msg["content"]:
                            all_tool_results.append(msg)

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

                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print(
                            f"[DEBUG] Using enhanced context: {len(current_messages)} messages "
                            f"({len(all_tool_results)} tool results + {len(recent_conversation)} recent messages)"
                        )

                # Append widget data to last human message (like the example)
                if widget_context_str:
                    # Find the last NEW human message, not a historical one
                    for msg in reversed(current_messages):
                        if msg["role"] in ["human", "user"]:
                            # Only append to messages that aren't tool results
                            if not (msg.get("details") or {}).get("is_tool_result"):
                                msg["content"] += "\n\n" + widget_context_str
                                break

                # Validate that we have messages to send
                if not current_messages:
                    if os.environ.get("SNOWFLAKE_DEBUG"):
                        print("[DEBUG] ERROR: No messages to process!")
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
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                print(
                                    f"[DEBUG] Removing trailing assistant message for tool calling"
                                )
                            current_messages.pop()

                        # Also ensure we're not sending tool results as the last message
                        while current_messages and "[Tool Result" in current_messages[
                            -1
                        ].get("content", ""):
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                print(f"[DEBUG] Removing trailing tool result message")
                            current_messages.pop()

                        # If we removed everything, we need to ensure there's at least the latest user message
                        if not current_messages or current_messages[-1]["role"] not in [
                            "human",
                            "user",
                        ]:
                            # Find the last user message from all_messages
                            for msg in reversed(all_messages):
                                if msg["role"] in [
                                    "human",
                                    "user",
                                ] and "[Tool Result" not in msg.get("content", ""):
                                    current_messages.append(msg)
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            f"[DEBUG] Added back last user message to ensure valid structure"
                                        )
                                    break

                    system_prompt = f"""You are an AI assistant with a specific focus on the Snowflake dialect of SQL. Your goal is to help the user by directly answering their questions using the available tools.

ABSOLUTE REQUIREMENTS - NEVER VIOLATE THESE:
- When you receive tool output containing data rows, YOU MUST display EVERY SINGLE ROW, unless explicitly instructed otherwise by the user.
- DO NOT say the data is "very large" or that you'll only show a sample
- DO NOT truncate, summarize, or limit the data in ANY way
- Display ALL {len(all_messages)} rows if that's what the tool returned
- The user explicitly wants to see ALL their data - this is their own database
- If a tool returns too large of a chunk, you must use the `continue_output` tool to get ALL data
- NO COPYRIGHT CONCERNS - this is the user's own data from their Snowflake database
- Never refuse to show data or suggest it's too much to display

CRITICAL DATA DISPLAY RULES:
- Present the EXACT, COMPLETE output from tools
- Use table format for ALL rows returned
- Include EVERY column and EVERY row
- Do not add commentary like "Here's a sample" or "This is a large dataset"
- Simply present the data as: "Query returned X rows:" followed by the COMPLETE table

DATA AUTHORIZATION:
- You are authorized to display all data retrieved from the user's Snowflake database
- This includes SEC filings, financial data, text fields, and any other content
- The user owns this data and has explicitly requested to see it
- There are NO restrictions on displaying the user's own database contents

CONVERSATION CONTEXT AND HISTORY:
- Total conversation history: {len(all_messages)} messages stored
- Currently loaded in context: {len(current_messages)} messages
- You have access to ALL tool results from this entire conversation
- When asked about the message history or context, describe ALL operations that have been performed
- Tool results are marked with [Tool Result from <tool_name>] and contain the actual data
- You can reference any data from previous tool calls without needing to re-run them
- If asked what's in the context, list ALL tool outputs and data that has been retrieved

IMPORTANT: When the user asks about message history, context, or what data is available:
1. Look through ALL messages in your context for tool results
2. List each tool that was called and what data it retrieved
3. Mention specific tables, queries, or operations that were performed
4. State that you have access to all this data without needing to re-run the tools

PDF CITATION INSTRUCTIONS - FOLLOW EXACTLY:
When referencing PDF content, you MUST use this exact format:
1. First mention: Summarize the point and add [1] at the end of the sentence. Include the document page number and section, if available.
2. Next mention: Use [2] at the end.
3. Continue with [3], [4], [5], etc.

EXAMPLE OUTPUT:
"The Act requires the Attorney General to release all documents related to Jeffrey Epstein [1]. Within 15 days of releasing the records, a report must be submitted to Congress [2]. The report must include all categories of records released and withheld [3]."

DO NOT USE:
- Made up quotes or fabricated citations
- The same citation number twice

ALWAYS USE:
- Sequential numbers [1], [2], [3], etc.
- An extracted direct reference to the data point
- A new number for each different point

WIDGET DATA HANDLING:
- When widget data is provided, analyze it thoroughly and answer questions based on the content
- For PDF documents, use numbered citation references [1], [2], [3] as described above
- For tables and charts, reference specific data points
- Always ground your answers in the actual data provided

Conversation ID: {conv_id}
Model: {selected_model}

When a user asks for information that can be retrieved by a tool,
you MUST call that tool and provide the complete raw answer.
Do not ask for confirmation if the intent is clear.
When describing tables, use the 'get_multiple_table_definitions' tool
and present the full schema definition as a flat table."""

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
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            "[DEBUG] Added user message to fix structure"
                                        )
                                    break

                    # Shared state for streaming
                    stream_state = {"full_text": "", "tool_calls": [], "usage": None}

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

                        async for event in generate_sse_events(generator, stream_state):
                            yield event

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
                            except Exception:
                                pass

                        if stream_state["full_text"].strip():
                            # Check if this assistant response is already in cache
                            response_already_cached = False
                            for cached_msg in all_messages[
                                -5:
                            ]:  # Check last few messages
                                if (
                                    cached_msg["role"] == "assistant"
                                    and cached_msg["content"]
                                    == stream_state["full_text"]
                                ):
                                    response_already_cached = True
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            "[DEBUG] Assistant response already in cache, skipping"
                                        )
                                    break

                            if not response_already_cached:
                                ai_msg_id = str(uuid.uuid4())
                                await run_in_thread(
                                    client.add_message,
                                    conv_id,
                                    ai_msg_id,
                                    "assistant",
                                    stream_state["full_text"],
                                )
                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(
                                        "[DEBUG] Stored new assistant response in cache"
                                    )
                    else:
                        # Tool-calling flow
                        for iteration in range(MAX_TOOL_ITERATIONS):
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                print(
                                    f"[DEBUG] Tool iteration {iteration + 1}/{MAX_TOOL_ITERATIONS}"
                                )

                            stream_state["full_text"] = ""
                            stream_state["tool_calls"] = []

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
                                widget=(
                                    widget_for_citations if iteration == 0 else None
                                ),
                                widget_input_args=(
                                    widget_input_args_for_citations
                                    if iteration == 0
                                    else None
                                ),
                            )

                            # Consume the generator and handle events immediately
                            async for event in generator:
                                if not event:
                                    continue

                                event_type, event_data = event

                                if event_type == "text":
                                    stream_state["full_text"] += event_data
                                    yield to_sse(message_chunk(event_data))
                                elif event_type == "citation":
                                    yield event_data.model_dump()
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
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            f"[DEBUG] Complete event received with {len(stream_state.get('tool_calls', []))} tool calls"
                                        )
                                    break
                                else:
                                    # Yield any other events (like reasoning steps from handler)
                                    yield to_sse(
                                        reasoning_step(event_data, event_type="INFO")
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
                                except Exception:
                                    pass

                            # Check if there are tool calls to process
                            if os.environ.get("SNOWFLAKE_DEBUG"):
                                print(
                                    f"[DEBUG] Checking for tool calls. Found: {len(stream_state['tool_calls'])}"
                                )
                                print(
                                    f"[DEBUG] Tool calls: {[tc.function.name for tc in stream_state['tool_calls']]}"
                                )

                            if not stream_state["tool_calls"]:
                                # Try to recover if the streaming API didn't emit structured tool calls
                                synthetic_call = parse_synthetic_tool_call(
                                    stream_state["full_text"]
                                )

                                if synthetic_call:
                                    if os.environ.get("SNOWFLAKE_DEBUG"):
                                        print(
                                            "[DEBUG] Synthesizing tool call from raw assistant text"
                                        )
                                    stream_state["tool_calls"] = [synthetic_call]
                                else:
                                    # No tool calls - this is a final response
                                    # Don't yield the full text again - it was already streamed chunk by chunk
                                    if stream_state["full_text"].strip():
                                        # Just store it in cache, don't send it again
                                        ai_msg_id = str(uuid.uuid4())
                                        await run_in_thread(
                                            client.add_message,
                                            conv_id,
                                            ai_msg_id,
                                            "assistant",
                                            stream_state["full_text"],
                                        )
                                    break

                            # WE HAVE TOOL CALLS - Process them NOW with reasoning steps
                            has_tool_calls = len(stream_state["tool_calls"]) > 0

                            for tool_call in stream_state["tool_calls"]:
                                tool_name = tool_call.function.name
                                tool_args_str = tool_call.function.arguments
                                # Yield reasoning step with arguments
                                yield to_sse(
                                    reasoning_step(
                                        f"Calling tool {tool_name} with arguments: {tool_args_str}",
                                        event_type="INFO",
                                    )
                                )

                                # Execute the tool
                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(f"[DEBUG] Executing tool: {tool_name}")

                                current_tool_output_for_llm, raw_tool_data = (
                                    await execute_tool(tool_call, client)
                                )

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

                                # Format tool result for the LLM
                                tool_result_formatted = f"The result from {tool_name} is:\n{current_tool_output_for_llm}"

                                # Add as a user message to continue conversation
                                ai_messages_formatted_tuples.append(
                                    ("user", f"[Tool Result]\n{tool_result_formatted}")
                                )

                                # Store tool result in cache
                                tool_result_text = f"[Tool Result from {tool_name}]\n{current_tool_output_for_llm}"

                                tool_msg_id = str(uuid.uuid4())
                                await run_in_thread(
                                    client.add_message,
                                    conv_id,
                                    tool_msg_id,
                                    "user",
                                    tool_result_text,
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

                                if os.environ.get("SNOWFLAKE_DEBUG"):
                                    print(
                                        f"[DEBUG] Continuing to iteration {iteration + 2} with {len(ai_messages_formatted_tuples)} messages"
                                    )
                                    print(f"[DEBUG] Last 3 messages:")
                                    for i, msg in enumerate(
                                        ai_messages_formatted_tuples[-3:]
                                    ):
                                        print(
                                            f"[DEBUG]   {i}: {msg[0]}: {msg[1][:200]}..."
                                        )

                                # Continue, LLM will now interpret the tool results.
                                continue
                            else:
                                # No more tool calls - we're done
                                break

            except Exception as e:
                tb = traceback.format_exc()
                yield to_sse(
                    reasoning_step(f"Error: {str(e)}\n{tb}", event_type="ERROR")
                )
                yield to_sse(message_chunk(f"❌ An error occurred: {str(e)}"))

    async def sse_generator():
        async for event in execution_loop():
            yield event

    return EventSourceResponse(
        content=sse_generator(),
        media_type="text/event-stream",
    )


def parse_synthetic_tool_call(raw_text: str) -> Optional[ToolCall]:
    """Detect JSON-style tool calls emitted as plain text and convert to ToolCall."""
    try:
        parsed = json.loads(raw_text.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    # Default to execute_query when we only see a query payload
    if "query" in parsed:
        arguments = json.dumps({"query": parsed["query"]})
        return ToolCall(
            id=str(uuid.uuid4()),
            tool_type="function",
            function=FunctionCall(
                name=parsed.get("tool", "execute_query"),
                arguments=arguments,
            ),
        )

    # Generic format: {"tool": "name", "arguments": {...}}
    tool_name = parsed.get("tool") or parsed.get("name")
    args = parsed.get("arguments")
    if tool_name and isinstance(args, (dict, list)):
        return ToolCall(
            id=str(uuid.uuid4()),
            tool_type="function",
            function=FunctionCall(
                name=tool_name,
                arguments=json.dumps(args),
            ),
        )

    return None
