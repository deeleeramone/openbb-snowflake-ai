"""Intelligent context management for Snowflake AI.

This module manages the context window to ensure the LLM can properly
ingest all information without exceeding token limits.
"""

from typing import Any
from .logger import get_logger

logger = get_logger(__name__)

# Cortex API limit is 32768 tokens
# We need to leave room for:
# - System prompt (~2000 tokens)
# - Current user query + widget context (~2000 tokens)
# - Model response (~4000 tokens max_tokens default)
# So available for conversation history + tool results: ~24000 tokens
MAX_CONTEXT_TOKENS = 32768
RESERVED_FOR_SYSTEM = 2500
RESERVED_FOR_CURRENT_QUERY = 3000
RESERVED_FOR_RESPONSE = 4500
AVAILABLE_FOR_HISTORY = (
    MAX_CONTEXT_TOKENS
    - RESERVED_FOR_SYSTEM
    - RESERVED_FOR_CURRENT_QUERY
    - RESERVED_FOR_RESPONSE
)

# Limits for different content types
MAX_SINGLE_TOOL_RESULT_TOKENS = 8000  # Single tool result shouldn't exceed this
MAX_TOTAL_TOOL_RESULTS_TOKENS = 12000  # All tool results combined
MAX_CONVERSATION_TOKENS = 8000  # Regular conversation history


def estimate_tokens(text: str) -> int:
    """Estimate token count - roughly 4 characters per token for English."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def summarize_tool_result(content: str, tool_name: str, max_tokens: int = 2000) -> str:
    """
    Intelligently summarize a tool result to fit within token limits.

    For document content, extracts key sections.
    For data results, keeps structure but truncates rows.
    """
    current_tokens = estimate_tokens(content)

    if current_tokens <= max_tokens:
        return content

    # Target character count (4 chars per token estimate)
    target_chars = max_tokens * 4

    # For read_document results, extract key sections intelligently
    if tool_name == "read_document" or "pages" in content.lower():
        return _summarize_document_content(content, target_chars)

    # For search results, keep the most relevant matches
    if tool_name == "search_document" or "search result" in content.lower():
        return _summarize_search_results(content, target_chars)

    # For query results, keep structure and sample rows
    if tool_name in ("execute_query", "get_table_sample_data") or "rowData" in content:
        return _summarize_query_results(content, target_chars)

    # Generic summarization - keep beginning and end with truncation notice
    return _generic_truncate(content, target_chars, tool_name)


def _summarize_document_content(content: str, target_chars: int) -> str:
    """Summarize document content by keeping key sections."""
    lines = content.split("\n")

    # Try to identify page boundaries
    pages = []
    current_page = []
    current_page_num = None

    for line in lines:
        # Detect page markers like "--- Page 1 ---" or "Page 1:"
        if line.strip().startswith("---") and "Page" in line:
            if current_page:
                pages.append((current_page_num, "\n".join(current_page)))
            current_page = []
            try:
                # Extract page number
                parts = line.split("Page")
                if len(parts) > 1:
                    num_part = parts[1].strip().split()[0].replace("-", "").strip()
                    current_page_num = (
                        int(num_part) if num_part.isdigit() else len(pages) + 1
                    )
            except (ValueError, IndexError):
                current_page_num = len(pages) + 1
        else:
            current_page.append(line)

    if current_page:
        pages.append((current_page_num, "\n".join(current_page)))

    if not pages:
        # No page structure found, use generic truncation
        return _generic_truncate(content, target_chars, "document")

    # Calculate how much we can keep per page
    num_pages = len(pages)
    chars_per_page = target_chars // min(num_pages, 10)  # Limit to 10 pages

    summarized_parts = []
    total_chars = 0
    pages_included = 0

    # Prioritize first few pages and last page
    priority_pages = []
    if num_pages <= 10:
        priority_pages = list(range(num_pages))
    else:
        # First 5, last 2, and evenly spaced middle pages
        priority_pages = list(range(5))  # First 5
        priority_pages.extend([num_pages - 2, num_pages - 1])  # Last 2
        # Add some middle pages
        step = (num_pages - 7) // 3
        for i in range(5, num_pages - 2, step):
            if i not in priority_pages:
                priority_pages.append(i)
        priority_pages = sorted(set(priority_pages))[:10]

    for idx in priority_pages:
        if idx >= len(pages):
            continue
        page_num, page_content = pages[idx]

        # Truncate each page if needed
        if len(page_content) > chars_per_page:
            page_content = page_content[:chars_per_page] + "\n[... page truncated ...]"

        summarized_parts.append(f"--- Page {page_num} ---\n{page_content}")
        total_chars += len(page_content)
        pages_included += 1

        if total_chars >= target_chars:
            break

    if pages_included < num_pages:
        summarized_parts.append(
            f"\n[... {num_pages - pages_included} additional pages omitted for context efficiency ...]"
        )

    return "\n\n".join(summarized_parts)


def _summarize_search_results(content: str, target_chars: int) -> str:
    """Summarize search results by keeping the most relevant matches."""
    lines = content.split("\n")

    # Try to identify result blocks
    results = []
    current_result = []

    for line in lines:
        # Detect result markers
        if line.strip().startswith(("Result ", "Match ", "---", "Page ", "Score:")):
            if current_result:
                results.append("\n".join(current_result))
            current_result = [line]
        else:
            current_result.append(line)

    if current_result:
        results.append("\n".join(current_result))

    if not results or len(results) <= 1:
        return _generic_truncate(content, target_chars, "search")

    # Keep top results within budget
    summarized = []
    total_chars = 0

    for result in results:
        if total_chars + len(result) > target_chars:
            # Truncate this result
            remaining = target_chars - total_chars
            if remaining > 200:
                summarized.append(result[:remaining] + "\n[... truncated ...]")
            break
        summarized.append(result)
        total_chars += len(result)

    if len(summarized) < len(results):
        summarized.append(
            f"\n[... {len(results) - len(summarized)} additional results omitted ...]"
        )

    return "\n\n".join(summarized)


def _summarize_query_results(content: str, target_chars: int) -> str:
    """Summarize query results by keeping schema and sample rows."""
    # Try to preserve column headers and first/last rows
    lines = content.split("\n")

    if len(lines) <= 20:
        return _generic_truncate(content, target_chars, "query")

    # Keep header (first few lines) and sample rows
    header_lines = lines[:5]
    footer_lines = lines[-3:]

    # Calculate remaining budget
    header_chars = sum(len(line) for line in header_lines)
    footer_chars = sum(len(line) for line in footer_lines)
    remaining_chars = target_chars - header_chars - footer_chars - 100  # Buffer

    # Keep as many middle rows as we can
    middle_lines = lines[5:-3]
    kept_middle = []
    chars_used = 0

    for line in middle_lines:
        if chars_used + len(line) > remaining_chars:
            break
        kept_middle.append(line)
        chars_used += len(line)

    result_lines = header_lines + kept_middle
    if len(kept_middle) < len(middle_lines):
        result_lines.append(
            f"[... {len(middle_lines) - len(kept_middle)} rows omitted ...]"
        )
    result_lines.extend(footer_lines)

    return "\n".join(result_lines)


def _generic_truncate(content: str, target_chars: int, content_type: str) -> str:
    """Generic truncation keeping beginning and end."""
    if len(content) <= target_chars:
        return content

    # Keep 70% from beginning, 20% from end
    begin_chars = int(target_chars * 0.7)
    end_chars = int(target_chars * 0.2)

    beginning = content[:begin_chars]
    ending = content[-end_chars:] if end_chars > 0 else ""

    omitted_chars = len(content) - begin_chars - end_chars

    return f"{beginning}\n\n[... {omitted_chars:,} characters ({omitted_chars // 4:,} tokens) omitted from {content_type} for context efficiency ...]\n\n{ending}"


def compress_conversation_history(
    messages: list[dict[str, Any]],
    current_query: str,
    max_tokens: int = AVAILABLE_FOR_HISTORY,
) -> list[dict[str, Any]]:
    """
    Compress conversation history to fit within token limits while preserving
    the most relevant context for the current query.

    Strategy:
    1. Always keep the current user query (unmodified)
    2. Keep recent tool results but summarize them if large
    3. Summarize older conversation turns
    4. Prioritize tool results that seem relevant to current query
    """
    if not messages:
        return messages

    # Separate messages by type
    tool_results = []
    regular_messages = []

    for msg in messages:
        content = msg.get("content", "")
        if "[Tool Result" in content:
            # Extract tool name for smart summarization
            tool_name = "unknown"
            if "[Tool Result from " in content:
                try:
                    tool_name = content.split("[Tool Result from ")[1].split("]")[0]
                except (IndexError, AttributeError):
                    pass
            tool_results.append({**msg, "_tool_name": tool_name})
        else:
            regular_messages.append(msg)

    # Budget allocation
    tool_budget = min(MAX_TOTAL_TOOL_RESULTS_TOKENS, max_tokens * 0.6)
    conversation_budget = max_tokens - tool_budget

    # Process tool results - summarize large ones, prioritize recent
    compressed_tool_results = []
    tool_tokens_used = 0

    # Process in reverse (most recent first)
    for tool_msg in reversed(tool_results):
        content = tool_msg.get("content", "")
        tool_name = tool_msg.get("_tool_name", "unknown")

        content_tokens = estimate_tokens(content)

        # Check if this single result is too large
        if content_tokens > MAX_SINGLE_TOOL_RESULT_TOKENS:
            # Summarize this tool result
            content = summarize_tool_result(
                content, tool_name, MAX_SINGLE_TOOL_RESULT_TOKENS
            )
            content_tokens = estimate_tokens(content)

        # Check if we have budget
        if tool_tokens_used + content_tokens <= tool_budget:
            compressed_msg = {**tool_msg, "content": content}
            del compressed_msg["_tool_name"]
            compressed_tool_results.append(compressed_msg)
            tool_tokens_used += content_tokens
        else:
            # Try to fit a summarized version
            remaining_budget = int(tool_budget - tool_tokens_used)
            if remaining_budget > 500:  # Only if we have meaningful space
                summarized = summarize_tool_result(content, tool_name, remaining_budget)
                compressed_msg = {**tool_msg, "content": summarized}
                del compressed_msg["_tool_name"]
                compressed_tool_results.append(compressed_msg)
            break  # No more budget

    # Reverse back to original order
    compressed_tool_results.reverse()

    # Process regular conversation - keep recent, summarize old
    compressed_conversation = []
    conversation_tokens_used = 0

    # Always include the last few messages in full
    KEEP_RECENT = 6
    recent_messages = (
        regular_messages[-KEEP_RECENT:]
        if len(regular_messages) > KEEP_RECENT
        else regular_messages
    )
    older_messages = (
        regular_messages[:-KEEP_RECENT] if len(regular_messages) > KEEP_RECENT else []
    )

    # Add recent messages
    for msg in recent_messages:
        content = msg.get("content", "")
        tokens = estimate_tokens(content)
        if conversation_tokens_used + tokens <= conversation_budget:
            compressed_conversation.append(msg)
            conversation_tokens_used += tokens

    # Add older messages if we have budget, but summarize them
    if older_messages and conversation_tokens_used < conversation_budget * 0.7:
        remaining_budget = conversation_budget - conversation_tokens_used

        # Create a summary of older conversation
        older_summary_parts = []
        for msg in older_messages[-10:]:  # Last 10 older messages
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate each message
            if len(content) > 200:
                content = content[:200] + "..."
            older_summary_parts.append(f"[{role}]: {content}")

        if older_summary_parts:
            older_summary = "[Earlier conversation context]\n" + "\n".join(
                older_summary_parts
            )
            if estimate_tokens(older_summary) <= remaining_budget:
                compressed_conversation.insert(
                    0,
                    {
                        "role": "system",
                        "content": older_summary,
                        "details": {"is_summary": True},
                    },
                )

    # Merge tool results back into conversation in original order
    # This is a simplified merge - in practice you'd want to preserve original ordering
    final_messages = []

    # Simple approach: tool results first (as context), then conversation
    final_messages.extend(compressed_tool_results)
    final_messages.extend(compressed_conversation)

    return final_messages


def prepare_messages_for_api(
    messages: list[tuple[str, str]],
    system_prompt: str,
    max_total_tokens: int = MAX_CONTEXT_TOKENS - RESERVED_FOR_RESPONSE,
) -> list[tuple[str, str]]:
    """
    Final preparation of messages before sending to API.
    Ensures total token count is within limits.

    Args:
        messages: List of (role, content) tuples
        system_prompt: The system prompt (already included in messages usually)
        max_total_tokens: Maximum tokens allowed

    Returns:
        Compressed messages list
    """
    # Calculate current token usage
    total_tokens = sum(estimate_tokens(content) for _, content in messages)

    if total_tokens <= max_total_tokens:
        return messages

    logger.warning(
        "Message context exceeds limit: %d tokens > %d max. Compressing...",
        total_tokens,
        max_total_tokens,
    )

    # Need to compress - start by identifying message types
    system_msgs = [(i, r, c) for i, (r, c) in enumerate(messages) if r == "system"]

    # Compress strategy:
    # 1. Keep system prompt but maybe truncate if huge
    # 2. Summarize large tool results in user messages
    # 3. Keep recent exchanges, summarize older ones

    compressed = []
    tokens_used = 0

    # Add system (allow up to 25% of budget)
    system_budget = int(max_total_tokens * 0.25)
    for _, role, content in system_msgs:
        if estimate_tokens(content) > system_budget:
            content = (
                content[: system_budget * 4] + "\n[... system prompt truncated ...]"
            )
        compressed.append((role, content))
        tokens_used += estimate_tokens(content)

    # Process non-system messages
    non_system = [(i, r, c) for i, (r, c) in enumerate(messages) if r != "system"]

    # Always keep the last user message and recent context
    if non_system:
        # Process from most recent to oldest
        for i, role, content in reversed(non_system):
            content_tokens = estimate_tokens(content)

            if tokens_used + content_tokens <= max_total_tokens:
                compressed.append((role, content))
                tokens_used += content_tokens
            else:
                # Try to summarize
                available = max_total_tokens - tokens_used
                if available > 200:
                    if "[Tool Result" in content:
                        # Extract tool name
                        tool_name = "unknown"
                        if "[Tool Result from " in content:
                            try:
                                tool_name = content.split("[Tool Result from ")[
                                    1
                                ].split("]")[0]
                            except (IndexError, AttributeError):
                                pass
                        summarized = summarize_tool_result(
                            content, tool_name, available
                        )
                    else:
                        summarized = _generic_truncate(content, available * 4, role)
                    compressed.append((role, summarized))
                    tokens_used += estimate_tokens(summarized)
                break  # Stop adding messages

    # Sort back to original order (by reconstructing based on position)
    # Since we processed system first, then reversed non-system, we need to fix order
    final = []
    system_added = False
    for role, content in messages:
        if role == "system" and not system_added:
            for r, c in compressed:
                if r == "system":
                    final.append((r, c))
            system_added = True
        elif role != "system":
            # Find matching compressed message
            for r, c in compressed:
                if r == role and (c == content or c.startswith(content[:50])):
                    if (r, c) not in final:
                        final.append((r, c))
                    break

    # Fallback: if reconstruction failed, just return compressed as-is
    if len(final) < len(compressed):
        final = compressed

    final_tokens = sum(estimate_tokens(c) for _, c in final)
    logger.info("Compressed context from %d to %d tokens", total_tokens, final_tokens)

    return final
