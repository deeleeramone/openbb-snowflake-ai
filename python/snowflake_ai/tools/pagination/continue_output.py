"""Continue output tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_continue_output(ctx: "ToolContext", args: dict):
    """Handle continue_output tool call."""
    continuation_key = args.get("continuation_key", "")
    conv_id = args.get("conversation_id", ctx.conv_id)

    try:
        continuation_data = await asyncio.to_thread(
            ctx.client.get_conversation_data, conv_id, continuation_key
        )

        if not continuation_data:
            yield "No continuation data found.", {"error": "No continuation data"}
            return

        data = json.loads(continuation_data)
        headers = data["headers"]
        remaining_rows = data["remaining_rows"]
        start_row = data["start_row"]
        total_rows = data["total_rows"]

        output = f"Continuing rows {start_row}-{min(start_row + len(remaining_rows) - 1, total_rows)} of {total_rows}:\n\n"
        output += "| " + " | ".join(headers) + " |\n"
        output += "|" + "|".join(["---" for _ in headers]) + "|\n"

        for row in remaining_rows:
            values = [str(row.get(h, "")) for h in headers]
            output += "| " + " | ".join(values) + " |\n"

        yield output, {"rows_displayed": len(remaining_rows)}

    except Exception as e:
        error_msg = f"Error continuing output: {str(e)}"
        yield error_msg, {"error": str(e)}
