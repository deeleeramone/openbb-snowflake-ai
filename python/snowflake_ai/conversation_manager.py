"""Conversation history and context management for Snowflake AI."""

import os
import uuid
from typing import Any

from ._snowflake_ai import SnowflakeAI
from .helpers import parse_widget_data


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


async def load_conversation_history(
    client: SnowflakeAI,
    conv_id: str,
    request_messages: list,
    max_cached_messages: int = 20,
) -> list[dict[str, Any]]:
    """
    Load conversation history from cache and add new request messages.

    Args:
        client: SnowflakeAI client instance
        conv_id: Conversation ID
        request_messages: New messages from the current request
        max_cached_messages: Maximum number of cached messages to load

    Returns:
        List of conversation messages with role, content, and details
    """
    final_conversation_history = []
    widget_data_context = []  # Collect widget data for non-tool models

    # Load existing messages from cache
    cached_messages = await run_in_thread(client.get_messages, conv_id)

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] Loaded {len(cached_messages)} messages from cache for conversation {conv_id}"
        )

    # Limit how many cached messages we load
    if len(cached_messages) > max_cached_messages:
        cached_messages = cached_messages[-max_cached_messages:]
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Truncated cached messages to last {max_cached_messages}")

    # Build history from cached messages, filtering out tool messages
    for message_id, role, content in cached_messages:
        # Skip tool messages from cache - they shouldn't be in conversation history
        if role == "tool":
            continue

        final_conversation_history.append(
            {
                "role": role,
                "content": content,
                "details": None,
            }
        )

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] final_conversation_history now has {len(final_conversation_history)} messages"
        )

    # Add new messages from the current request
    for message in request_messages:
        msg_id = str(uuid.uuid4())

        msg_dict = {
            "role": message.role,
            "content": message.content,
            "details": None,
        }

        if message.role == "tool" and hasattr(message, "data"):
            context_str = parse_widget_data(message.data, conv_id)
            msg_dict["content"] = f"Tool output:\n{context_str}"
            msg_dict["details"] = message.data

            # Store widget data for potential injection
            widget_data_context.append({"data": message.data, "parsed": context_str})

        final_conversation_history.append(msg_dict)

        # Store message immediately
        await run_in_thread(
            client.add_message,
            conv_id,
            msg_id,
            message.role,
            msg_dict["content"],
        )

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] After adding new messages: {len(final_conversation_history)} total messages"
        )

    # Store widget data context for later use
    if widget_data_context:
        final_conversation_history.append(
            {
                "role": "widget_context",
                "content": widget_data_context,
                "details": None,
            }
        )

    return final_conversation_history


async def condense_conversation_context(
    final_conversation_history: list[dict[str, Any]],
    context_size_threshold: int = 10,
    messages_to_keep: int = 3,
) -> list[dict[str, Any]]:
    """
    Condense conversation history to prevent payload errors.

    Args:
        final_conversation_history: Full conversation history
        context_size_threshold: Threshold to trigger condensing
        messages_to_keep: Number of recent messages to keep

    Returns:
        Condensed conversation history
    """
    llm_context_messages = final_conversation_history

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] llm_context_messages has {len(llm_context_messages)} messages before condensing"
        )

    if len(llm_context_messages) > context_size_threshold:
        history_to_summarize = llm_context_messages[:-messages_to_keep]

        # Create a very brief summary
        summary_text = f"Previous conversation context: The user and assistant have been discussing various topics over {len(history_to_summarize)} messages."

        # Identify main topic
        recent_content = " ".join(
            [
                msg["content"][:100]
                for msg in history_to_summarize[-3:]
                if msg["content"]
            ]
        )
        if "table" in recent_content.lower():
            summary_text += (
                " The conversation has involved database tables and schemas."
            )
        elif "query" in recent_content.lower() or "sql" in recent_content.lower():
            summary_text += " The conversation has involved SQL queries."

        summary_message = {
            "role": "system",
            "content": summary_text,
            "details": None,
        }
        llm_context_messages = [summary_message] + llm_context_messages[
            -messages_to_keep:
        ]

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(
                f"[DEBUG] After condensing: {len(llm_context_messages)} messages in context"
            )

    return llm_context_messages


def format_messages_for_llm(
    messages: list[dict],
    system_prompt: str,
    inject_widget_data: bool = False,
) -> list[tuple[str, str]]:
    """
    Format messages for the LLM, including tool results as context.

    Args:
        messages: List of message dicts with 'role' and 'content'
        system_prompt: System prompt to prepend
        inject_widget_data: Whether to inject widget data (deprecated)

    Returns:
        List of (role, content) tuples formatted for the LLM
    """
    formatted = [("system", system_prompt)]

    # Track tool results to include them with context
    pending_tool_results = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Map roles to what the LLM expects
        if role in ["human", "user"]:
            # Check if this is a tool result
            if content.startswith("[Tool Result"):
                # Include tool results as user messages for context
                formatted.append(("user", content))
            else:
                # Regular user message
                formatted.append(("user", content))
        elif role in ["assistant", "ai"]:
            formatted.append(("assistant", content))
        elif role == "system":
            # Merge system messages
            if formatted and formatted[0][0] == "system":
                formatted[0] = ("system", formatted[0][1] + "\n\n" + content)
            else:
                formatted.append(("system", content))
        elif role == "tool":
            # Tool messages get converted to user messages for context
            formatted.append(("user", f"[Tool Output]\n{content}"))

    return formatted


async def store_message(
    client: SnowflakeAI,
    conv_id: str,
    role: str,
    content: str,
) -> str:
    """
    Store a message in the conversation cache.

    Args:
        client: SnowflakeAI client instance
        conv_id: Conversation ID
        role: Message role (human, assistant, tool, system)
        content: Message content

    Returns:
        Generated message ID
    """
    msg_id = str(uuid.uuid4())
    await run_in_thread(
        client.add_message,
        conv_id,
        msg_id,
        role,
        content,
    )
    return msg_id
