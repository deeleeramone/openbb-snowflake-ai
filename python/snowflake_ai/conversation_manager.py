"""Conversation history and context management for Snowflake AI."""

import asyncio
import os
import uuid
from typing import Any

from ._snowflake_ai import SnowflakeAI
from .helpers import parse_widget_data


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def load_conversation_history(
    client: SnowflakeAI,
    conv_id: str,
    request_messages: list,
) -> list[dict[str, Any]]:
    """
    Load conversation history from the agent's database and add new request messages.

    Args:
        client: SnowflakeAI client instance
        conv_id: Conversation ID
        request_messages: New messages from the current request

    Returns:
        List of conversation messages with role, content, and details
    """
    final_conversation_history = []

    # Load existing messages from the agent's database
    cached_messages = await run_in_thread(client.get_messages, conv_id)

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] Loaded {len(cached_messages)} messages from database for conversation {conv_id}"
        )

    # Build history from cached messages
    for message_id, role, content in cached_messages:
        final_conversation_history.append(
            {
                "role": role,
                "content": content,
                "details": {"message_id": message_id},
            }
        )

    # Process new messages from the current request
    for message in request_messages:
        # Check if message already exists
        msg_content = message.content if hasattr(message, "content") else str(message)
        is_duplicate = any(
            msg["role"] == message.role and msg["content"] == msg_content
            for msg in final_conversation_history
        )

        if not is_duplicate:
            msg_id = str(uuid.uuid4())
            msg_dict = {
                "role": message.role,
                "content": msg_content,
                "details": None,
            }

            if message.role == "tool" and hasattr(message, "data"):
                context_str = parse_widget_data(message.data, conv_id)
                msg_dict["content"] = f"Tool output:\n{context_str}"
                msg_dict["details"] = message.data

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
            f"[DEBUG] After processing: {len(final_conversation_history)} total messages"
        )

    return final_conversation_history


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

    # Track tool results to attach after assistant messages
    pending_tool_results: list[str] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Skip empty messages
        if not content or not content.strip():
            continue

        # Map roles to what the LLM expects
        if role in ["human", "user"]:
            # Flush any pending tool results before user message
            for tool_result in pending_tool_results:
                formatted.append(("user", tool_result))
            pending_tool_results.clear()

            # Check if this is a tool result
            if content.startswith("[Tool Result"):
                # Include tool results as user messages for context
                formatted.append(("user", content))
            else:
                # Regular user message
                formatted.append(("user", content))
        elif role in ["assistant", "ai"]:
            # Flush any pending tool results before assistant message
            for tool_result in pending_tool_results:
                formatted.append(("user", tool_result))
            pending_tool_results.clear()

            formatted.append(("assistant", content))
        elif role == "system":
            # Merge system messages
            if formatted and formatted[0][0] == "system":
                formatted[0] = ("system", formatted[0][1] + "\n\n" + content)
            else:
                formatted.append(("system", content))
        elif role == "tool":
            # Queue tool results to be added after the assistant message that called them
            pending_tool_results.append(f"[Tool Output]\n{content}")

    # Flush any remaining tool results at the end
    for tool_result in pending_tool_results:
        formatted.append(("user", tool_result))

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
