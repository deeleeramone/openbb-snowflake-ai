"""Cache management utilities for conversation history optimization."""

import asyncio
import os
from typing import Optional

from ._snowflake_ai import SnowflakeAI


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def get_conversation_summary(
    client: SnowflakeAI,
    conv_id: str,
    last_n_messages: int = 10,
) -> Optional[str]:
    """
    Get a summary of the conversation context.

    Args:
        client: SnowflakeAI client instance
        conv_id: Conversation ID
        last_n_messages: Number of recent messages to summarize

    Returns:
        Summary string or None if no messages
    """
    try:
        messages = await run_in_thread(client.get_messages, conv_id)

        if not messages:
            return None

        recent_messages = messages[-last_n_messages:]

        # Build a concise summary
        summary_parts = []

        # Track main topics discussed
        topics = set()
        for _, role, content in recent_messages:
            content_lower = content.lower()
            if "table" in content_lower or "schema" in content_lower:
                topics.add("database schemas")
            if "query" in content_lower or "select" in content_lower:
                topics.add("SQL queries")
            if "epstein" in content_lower:
                topics.add("Epstein files analysis")
            if "name" in content_lower or "email" in content_lower:
                topics.add("name/email extraction")

        if topics:
            summary_parts.append(f"Topics discussed: {', '.join(topics)}")

        # Get last significant query or operation
        for _, role, content in reversed(recent_messages):
            if "CREATE TABLE" in content.upper() or "SELECT" in content.upper():
                summary_parts.append(f"Recent operation: SQL query execution")
                break

        return ". ".join(summary_parts) if summary_parts else "General conversation"

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Error getting conversation summary: {e}")
        return None


async def get_conversation_stats(
    client: SnowflakeAI,
    conv_id: str,
) -> dict:
    """
    Get statistics about the conversation.

    Args:
        client: SnowflakeAI client instance
        conv_id: Conversation ID

    Returns:
        Dictionary with conversation statistics
    """
    try:
        messages = await run_in_thread(client.get_messages, conv_id)

        stats = {
            "total_messages": len(messages),
            "human_messages": 0,
            "assistant_messages": 0,
            "tool_results": 0,
            "unique_tools_used": set(),
        }

        for _, role, content in messages:
            if role in ["human", "user"]:
                if "[Tool Result" in content:
                    stats["tool_results"] += 1
                    # Extract tool name
                    import re

                    match = re.search(r"\[Tool Result from (\w+)\]", content)
                    if match:
                        stats["unique_tools_used"].add(match.group(1))
                else:
                    stats["human_messages"] += 1
            elif role in ["assistant", "ai"]:
                stats["assistant_messages"] += 1
            elif role == "tool":
                stats["tool_results"] += 1

        stats["unique_tools_used"] = list(stats["unique_tools_used"])

        return stats

    except Exception as e:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Error getting conversation stats: {e}")
        return {}
