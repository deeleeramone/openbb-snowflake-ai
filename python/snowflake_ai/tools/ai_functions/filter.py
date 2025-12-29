"""AI filter tool handler."""

import asyncio
import json
import os
import traceback
from typing import TYPE_CHECKING

from openbb_ai import reasoning_step

from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_ai_filter(ctx: "ToolContext", args: dict):
    """Handle ai_filter tool call."""
    predicate = args.get("predicate", "")
    text = args.get("text")
    query = args.get("query")
    column_name = args.get("column_name")
    image_stage_path = args.get("image_stage_path")

    if not predicate:
        yield "Error: predicate is required", {"error": "predicate required"}
        return

    try:
        # Case 1: Single text classification
        if text and not query and not image_stage_path:
            text_escaped = text.replace("'", "''")
            predicate_escaped = predicate.replace("'", "''")

            # Use PROMPT to combine predicate with text
            filter_sql = f"""
            SELECT AI_FILTER(PROMPT('{predicate_escaped}: {{0}}', '{text_escaped}')) AS RESULT
            """
            result = await asyncio.to_thread(ctx.client.execute_query, filter_sql)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if row_data:
                ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                output = "**AI Filter Result:**\n\n"
                output += f"**Predicate:** {predicate}\n"
                output += (
                    f"**Text:** {text[:200]}{'...' if len(text) > 200 else ''}\n\n"
                )
                output += f"**Result:** {'✅ TRUE' if ai_result else '❌ FALSE'}"
                yield output, {"result": ai_result, "predicate": predicate}
            else:
                yield "No result returned from AI_FILTER", {"error": "No result"}
            return

        # Case 2: Image classification
        elif image_stage_path:
            predicate_escaped = predicate.replace("'", "''")
            # Parse stage path into stage name and file path
            clean_path = image_stage_path.lstrip("@")
            if "/" in clean_path:
                parts = clean_path.split("/", 1)
                stage_name = f"@{parts[0]}"
                file_path = parts[1]
            else:
                stage_name = f"@{clean_path}"
                file_path = ""

            filter_sql = f"""
            SELECT AI_FILTER('{predicate_escaped}', TO_FILE('{stage_name}', '{file_path}')) AS RESULT
            """
            result = await asyncio.to_thread(ctx.client.execute_query, filter_sql)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if row_data:
                ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                output = "**AI Filter Result (Image):**\n\n"
                output += f"**Predicate:** {predicate}\n"
                output += f"**Image:** {image_stage_path}\n\n"
                output += f"**Result:** {'✅ TRUE' if ai_result else '❌ FALSE'}"
                yield output, {
                    "result": ai_result,
                    "predicate": predicate,
                    "image": image_stage_path,
                }
            else:
                yield "No result returned from AI_FILTER", {"error": "No result"}
            return

        # Case 3: Filter query results
        elif query and column_name:
            predicate_escaped = predicate.replace("'", "''")
            column_escaped = column_name.replace('"', '""')

            # Wrap the user's query and apply AI_FILTER
            filter_sql = f"""
            WITH user_query AS (
                {query}
            )
            SELECT *, AI_FILTER(PROMPT('{predicate_escaped}: {{0}}', "{column_escaped}")) AS AI_FILTER_RESULT
            FROM user_query
            """

            yield to_sse(
                reasoning_step(
                    f"Applying AI_FILTER to query results on column '{column_name}'...",
                    event_type="INFO",
                )
            )

            result = await asyncio.to_thread(ctx.client.execute_query, filter_sql)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if not row_data:
                yield "Query returned no rows to filter", {
                    "results": [],
                    "predicate": predicate,
                }
                return

            # Count TRUE vs FALSE
            true_count = sum(
                1
                for row in row_data
                if row.get("AI_FILTER_RESULT", row.get("ai_filter_result"))
            )
            false_count = len(row_data) - true_count

            # Format output
            output = "**AI Filter Results:**\n\n"
            output += f"**Predicate:** {predicate}\n"
            output += f"**Column Filtered:** {column_name}\n"
            output += f"**Total Rows:** {len(row_data)}\n"
            output += f"**TRUE:** {true_count} | **FALSE:** {false_count}\n\n"

            # Show table with results
            if row_data:
                headers = list(row_data[0].keys())
                output += "| " + " | ".join(headers) + " |\n"
                output += "|" + "|".join(["---" for _ in headers]) + "|\n"

                for row in row_data[:20]:  # Limit to 20 rows for display
                    values = []
                    for h in headers:
                        val = row.get(h, "")
                        if h.upper() == "AI_FILTER_RESULT":
                            val = "✅" if val else "❌"
                        values.append(str(val)[:50])  # Truncate long values
                    output += "| " + " | ".join(values) + " |\n"

                if len(row_data) > 20:
                    output += f"\n*... and {len(row_data) - 20} more rows*"

            yield output, {
                "results": row_data,
                "predicate": predicate,
                "true_count": true_count,
                "false_count": false_count,
                "total_rows": len(row_data),
            }
            return

        # Case 4: Simple predicate evaluation (no text provided - just evaluate the predicate)
        else:
            predicate_escaped = predicate.replace("'", "''")
            filter_sql = f"""
            SELECT AI_FILTER('{predicate_escaped}') AS RESULT
            """
            result = await asyncio.to_thread(ctx.client.execute_query, filter_sql)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if row_data:
                ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                output = "**AI Filter Result:**\n\n"
                output += f"**Question/Statement:** {predicate}\n\n"
                output += f"**Result:** {'✅ TRUE' if ai_result else '❌ FALSE'}"
                yield output, {"result": ai_result, "predicate": predicate}
            else:
                yield "No result returned from AI_FILTER", {"error": "No result"}
            return

    except Exception as e:
        error_msg = f"Error executing AI_FILTER: {str(e)}"
        logger.error("AI_FILTER error: %s", traceback.format_exc())
        yield error_msg, {"error": str(e)}
