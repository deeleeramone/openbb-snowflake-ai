"""Table schema tool handlers."""

import asyncio
import json
import os
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext

from ...logger import get_logger

logger = get_logger(__name__)


async def handle_get_table_sample_data(ctx: "ToolContext", args: dict):
    """Handle get_table_sample_data tool call."""
    table_name = args.get("table_name", "")
    try:
        result = await asyncio.to_thread(
            ctx.client.get_table_sample_data_rust, table_name
        )
        result_json = json.loads(result)

        row_data = result_json.get("rowData", [])
        num_rows = len(row_data)

        if num_rows == 0:
            output = f"No data found in table {table_name}."
        else:
            # Limit rows for sample
            limit = 5
            headers = list(row_data[0].keys()) if row_data else []
            output = f"Sample data from {table_name} ({min(num_rows, limit)} of {num_rows} rows):\n\n"
            output += "| " + " | ".join(headers) + " |\n"
            output += "|" + "|".join(["---" for _ in headers]) + "|\n"

            for i, row in enumerate(row_data):
                if i >= limit:
                    break
                values = [str(row.get(h, "")) for h in headers]
                output += "| " + " | ".join(values) + " |\n"

        yield output, result_json

    except Exception as e:
        error_msg = f"Error getting sample data for {table_name}: {str(e)}"
        yield error_msg, {"error": str(e)}


async def handle_get_table_schema(ctx: "ToolContext", args: dict):
    """Handle get_table_schema tool call."""
    table_name = args.get("table_name", "")
    try:
        result = await asyncio.to_thread(ctx.client.get_table_info, table_name)
        result_json = json.loads(result)

        # Extract columns list from the result dict
        columns = result_json.get("columns", [])
        if not isinstance(columns, list):
            yield (
                f"Error: Invalid format for table schema for {table_name}.",
                {"error": "Invalid format"},
            )
            return

        output = f"Table schema for {table_name}:\n\n"
        output += "| Column Name | Data Type | Nullable | Default |\n"
        output += "|---|---|---|---|\n"

        for col in columns:
            output += f"| {col.get('COLUMN_NAME', '')} "
            output += f"| {col.get('DATA_TYPE', '')} "
            output += f"| {col.get('IS_NULLABLE', '')} "
            output += f"| {col.get('COLUMN_DEFAULT', '')} |\n"

        yield output, result_json

    except Exception as e:
        error_msg = f"Error getting table schema for {table_name}: {str(e)}"
        yield error_msg, {"error": str(e)}


async def handle_get_multiple_table_definitions(ctx: "ToolContext", args: dict):
    """Handle get_multiple_table_definitions tool call."""
    table_names = args.get("table_names", [])
    all_definitions = []
    combined_output = ""

    for table_name in table_names:
        try:
            result = await asyncio.to_thread(ctx.client.get_table_info, table_name)

            # Handle both string and dict responses
            if isinstance(result, str):
                try:
                    result_json = json.loads(result)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat it as an error
                    combined_output += (
                        f"\n### {table_name}\nError: Invalid response format\n"
                    )
                    continue
            else:
                result_json = result

            # Extract columns from the response
            # The Rust engine returns {"columns": [...], "primary_keys": [...], ...}
            if isinstance(result_json, dict) and "columns" in result_json:
                columns = result_json["columns"]
            elif isinstance(result_json, list):
                columns = result_json
            else:
                combined_output += f"\n### {table_name}\nError: Expected columns data, got {type(result_json).__name__}: {result_json}\n"
                continue

            all_definitions.append({"table": table_name, "columns": columns})

            # Add to output
            combined_output += f"\n### {table_name}\n"
            combined_output += "| Column Name | Data Type | Nullable | Default |\n"
            combined_output += "|---|---|---|---|\n"

            for col in columns:
                if isinstance(col, dict):
                    # Handle both naming conventions from Rust (COLUMN_NAME/DATA_TYPE or name/type)
                    col_name = col.get("COLUMN_NAME", col.get("name", ""))
                    data_type = col.get("DATA_TYPE", col.get("type", ""))
                    nullable = col.get("IS_NULLABLE", col.get("null?", ""))
                    default = col.get("COLUMN_DEFAULT", col.get("default", ""))
                    combined_output += (
                        f"| {col_name} | {data_type} | {nullable} | {default} |\n"
                    )
                else:
                    combined_output += f"| Column data is not a dictionary: {col} |\n"

        except Exception as e:
            combined_output += f"\n### {table_name}\nError: {str(e)}\n"
            logger.error(
                "Error getting table info for %s: %s",
                table_name,
                traceback.format_exc(),
            )

    yield combined_output.strip(), all_definitions
