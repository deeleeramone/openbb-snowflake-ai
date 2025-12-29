"""Query validation tool handler."""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_validate_query(ctx: "ToolContext", args: dict):
    """Handle validate_query tool call."""
    query = args.get("query", "")
    try:
        await asyncio.to_thread(ctx.client.validate_query, query)
        output = "Query is valid."
        yield output, {"valid": True}
    except Exception as e:
        error_msg = f"Query validation failed: {str(e)}"
        yield error_msg, {"valid": False, "error": str(e)}
