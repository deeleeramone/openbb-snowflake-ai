"""AI aggregation tool handlers."""

import asyncio
import json
import re
import traceback
from typing import TYPE_CHECKING

from openbb_ai import reasoning_step

from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_ai_agg(ctx: "ToolContext", args: dict):
    """Handle ai_agg tool call."""
    instruction = args.get("instruction", "")
    text = args.get("text")
    query = args.get("query")
    text_column = args.get("text_column")

    # Validate parameters
    if not instruction:
        error_msg = "Error: 'instruction' parameter is required for ai_agg"
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Missing instruction parameter"}
        return

    if not text and not (query and text_column):
        error_msg = "Error: Provide either 'text' for direct aggregation or both 'query' and 'text_column' for query-based aggregation"
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Missing required parameters"}
        return

    try:
        # Escape instruction for SQL
        instruction_escaped = instruction.replace("'", "''")

        # Mode 1: Direct text aggregation
        if text:
            text_escaped = text.replace("'", "''")
            ai_agg_sql = f"SELECT AI_AGG('{text_escaped}', '{instruction_escaped}') AS AGGREGATED_RESULT"

            yield to_sse(
                reasoning_step(
                    f"Applying AI_AGG to direct text with instruction: '{instruction[:50]}...'",
                    event_type="INFO",
                )
            )

        # Mode 2: Query-based aggregation
        else:
            text_col_escaped = text_column.replace('"', '""')

            # Check if query has GROUP BY
            has_group_by = bool(re.search(r"\bGROUP\s+BY\b", query, re.IGNORECASE))

            if has_group_by:
                # Query already has GROUP BY, wrap it and apply AI_AGG
                ai_agg_sql = f"""
                WITH source_data AS (
                    {query}
                )
                SELECT *, AI_AGG("{text_col_escaped}", '{instruction_escaped}') AS AGGREGATED_RESULT
                FROM source_data
                """
                yield to_sse(
                    reasoning_step(
                        f"Applying AI_AGG with GROUP BY aggregation: '{instruction[:50]}...'",
                        event_type="INFO",
                    )
                )
            else:
                # No GROUP BY, aggregate all rows
                ai_agg_sql = f"""
                WITH source_data AS (
                    {query}
                )
                SELECT AI_AGG("{text_col_escaped}", '{instruction_escaped}') AS AGGREGATED_RESULT
                FROM source_data
                """
                yield to_sse(
                    reasoning_step(
                        f"Applying AI_AGG to all rows with instruction: '{instruction[:50]}...'",
                        event_type="INFO",
                    )
                )

        # Execute query
        result = await asyncio.to_thread(ctx.client.execute_query, ai_agg_sql)
        result_json = json.loads(result)
        row_data = result_json.get("rowData", [])

        if not row_data:
            yield "No results from AI_AGG", {"error": "No results"}
            return

        # Format output
        output = f"**AI Aggregation Results**\n\n**Instruction:** {instruction}\n\n"

        if len(row_data) == 1:
            # Single result (no grouping)
            aggregated = row_data[0].get("AGGREGATED_RESULT") or row_data[0].get(
                "aggregated_result"
            )
            output += f"{aggregated}\n"

        else:
            # Multiple results (grouped)
            # Detect group columns (all columns except AGGREGATED_RESULT)
            group_cols = [
                col for col in row_data[0].keys() if col.upper() != "AGGREGATED_RESULT"
            ]

            for row in row_data:
                # Build hierarchical header for group
                for i, col in enumerate(group_cols):
                    indent = "  " * i
                    col_value = row.get(col)
                    output += f"{indent}**{col}:** {col_value}\n"

                # Add aggregation result
                aggregated = row.get("AGGREGATED_RESULT") or row.get(
                    "aggregated_result"
                )
                indent = "  " * len(group_cols)
                output += f"{indent}{aggregated}\n\n"

        yield output, {
            "instruction": instruction,
            "row_count": len(row_data),
            "has_grouping": len(row_data) > 1,
        }

    except Exception as e:
        error_msg = f"Error executing AI_AGG: {str(e)}"
        logger.error("AI_AGG error: %s", traceback.format_exc())
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": str(e)}


async def handle_ai_summarize_agg(ctx: "ToolContext", args: dict):
    """Handle ai_summarize_agg tool call."""
    text = args.get("text")
    query = args.get("query")
    text_column = args.get("text_column")

    # Validate parameters
    if not text and not (query and text_column):
        error_msg = "Error: Provide either 'text' for direct summarization or both 'query' and 'text_column' for query-based summarization"
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Missing required parameters"}
        return

    try:
        # Mode 1: Direct text summarization
        if text:
            text_escaped = text.replace("'", "''")
            summarize_sql = f"SELECT AI_SUMMARIZE_AGG('{text_escaped}') AS SUMMARY"

            yield to_sse(
                reasoning_step(
                    "Applying AI_SUMMARIZE_AGG to direct text",
                    event_type="INFO",
                )
            )

        # Mode 2: Query-based summarization
        else:
            text_col_escaped = text_column.replace('"', '""')

            # Check if query has GROUP BY
            has_group_by = bool(re.search(r"\bGROUP\s+BY\b", query, re.IGNORECASE))

            if has_group_by:
                # Query already has GROUP BY, wrap it and apply AI_SUMMARIZE_AGG
                summarize_sql = f"""
                WITH source_data AS (
                    {query}
                )
                SELECT *, AI_SUMMARIZE_AGG("{text_col_escaped}") AS SUMMARY
                FROM source_data
                """
                yield to_sse(
                    reasoning_step(
                        "Applying AI_SUMMARIZE_AGG with GROUP BY aggregation",
                        event_type="INFO",
                    )
                )
            else:
                # No GROUP BY, summarize all rows
                summarize_sql = f"""
                WITH source_data AS (
                    {query}
                )
                SELECT AI_SUMMARIZE_AGG("{text_col_escaped}") AS SUMMARY
                FROM source_data
                """
                yield to_sse(
                    reasoning_step(
                        "Applying AI_SUMMARIZE_AGG to all rows",
                        event_type="INFO",
                    )
                )

        # Execute query
        result = await asyncio.to_thread(ctx.client.execute_query, summarize_sql)
        result_json = json.loads(result)
        row_data = result_json.get("rowData", [])

        if not row_data:
            yield "No results from AI_SUMMARIZE_AGG", {"error": "No results"}
            return

        # Format output
        output = "**AI Summary**\n\n"

        if len(row_data) == 1:
            # Single result (no grouping)
            summary = row_data[0].get("SUMMARY") or row_data[0].get("summary")
            output += f"{summary}\n"

        else:
            # Multiple results (grouped)
            # Detect group columns (all columns except SUMMARY)
            group_cols = [col for col in row_data[0].keys() if col.upper() != "SUMMARY"]

            for row in row_data:
                # Build hierarchical header for group
                for i, col in enumerate(group_cols):
                    indent = "  " * i
                    col_value = row.get(col)
                    output += f"{indent}**{col}:** {col_value}\n"

                # Add summary
                summary = row.get("SUMMARY") or row.get("summary")
                indent = "  " * len(group_cols)
                output += f"{indent}{summary}\n\n"

        yield output, {
            "row_count": len(row_data),
            "has_grouping": len(row_data) > 1,
        }

    except Exception as e:
        error_msg = f"Error executing AI_SUMMARIZE_AGG: {str(e)}"
        logger.error("AI_SUMMARIZE_AGG error: %s", traceback.format_exc())
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": str(e)}
