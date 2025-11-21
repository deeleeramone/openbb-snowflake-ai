"""Tool executor for Snowflake AI server."""

import json
import os
from typing import Any, Tuple

from ._snowflake_ai import SnowflakeAI, ToolCall


async def execute_tool(
    tool_call: ToolCall,
    client: SnowflakeAI,
) -> Tuple[str, Any]:
    """
    Execute a tool call and return formatted output for both LLM and user.

    CRITICAL: This function must NEVER truncate, summarize, or hide data.
    ALL data from Snowflake must be returned in full.

    Returns:
        Tuple of (llm_output, raw_data)
        - llm_output: The COMPLETE data to send to the LLM
        - raw_data: The raw data for storage/caching
    """
    import asyncio

    tool_name = tool_call.function.name
    tool_args_str = tool_call.function.arguments

    try:
        tool_args = json.loads(tool_args_str) if tool_args_str else {}
    except json.JSONDecodeError:
        tool_args = {"raw": tool_args_str}

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(f"[DEBUG] Executing tool: {tool_name} with args: {tool_args}")

    # Execute the appropriate tool
    if tool_name == "get_table_sample_data":
        table_name = tool_args.get("table_name", "")
        try:
            result = await asyncio.to_thread(
                client.get_table_sample_data_rust, table_name
            )
            result_json = json.loads(result)

            # NEVER TRUNCATE - Return ALL data
            row_data = result_json.get("rowData", [])
            num_rows = len(row_data)

            if num_rows == 0:
                output = f"No data found in table {table_name}."
            else:
                # Format ALL rows as a table - NO TRUNCATION
                headers = list(row_data[0].keys()) if row_data else []
                output = f"Sample data from {table_name} ({num_rows} rows):\n\n"
                output += "| " + " | ".join(headers) + " |\n"
                output += "|" + "|".join(["---" for _ in headers]) + "|\n"

                # Include EVERY SINGLE ROW
                for row in row_data:
                    values = [str(row.get(h, "")) for h in headers]
                    output += "| " + " | ".join(values) + " |\n"

            return output, result_json

        except Exception as e:
            error_msg = f"Error getting sample data for {table_name}: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "get_table_definition":
        table_name = tool_args.get("table_name", "")
        try:
            result = await asyncio.to_thread(client.get_table_info, table_name)
            result_json = json.loads(result)

            # Handle cases where result_json is not a list of dicts
            if not isinstance(result_json, list) or not all(
                isinstance(item, dict) for item in result_json
            ):
                return (
                    f"Error: Invalid format for table definition for {table_name}. Expected a list of columns.",
                    {"error": "Invalid format"},
                )

            # Format ALL columns - NO TRUNCATION
            output = f"Table definition for {table_name}:\n\n"
            output += "| name | type | kind | null? | default | primary key | unique key | check | expression | comment | policy name | privacy domain |\n"
            output += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"

            for col in result_json:
                output += f"| {col.get('name', '')} "
                output += f"| {col.get('type', '')} "
                output += f"| {col.get('kind', '')} "
                output += f"| {col.get('null?', '')} "
                output += f"| {col.get('default', '')} "
                output += f"| {col.get('primary key', '')} "
                output += f"| {col.get('unique key', '')} "
                output += f"| {col.get('check', '')} "
                output += f"| {col.get('expression', '')} "
                output += f"| {col.get('comment', '')} "
                output += f"| {col.get('policy name', '')} "
                output += f"| {col.get('privacy domain', '')} |\n"

            return output, result_json

        except Exception as e:
            error_msg = f"Error getting table definition for {table_name}: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "get_multiple_table_definitions":
        table_names = tool_args.get("table_names", [])
        all_definitions = []
        combined_output = ""

        for table_name in table_names:
            try:
                result = await asyncio.to_thread(client.get_table_info, table_name)

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

                # Ensure result_json is a list
                if not isinstance(result_json, list):
                    combined_output += f"\n### {table_name}\nError: Expected list of columns, got {type(result_json).__name__}\n"
                    continue

                all_definitions.append({"table": table_name, "columns": result_json})

                # Add to output
                combined_output += f"\n### {table_name}\n"
                combined_output += "| name | type | kind | null? | default | primary key | unique key | check | expression | comment | policy name | privacy domain |\n"
                combined_output += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"

                for col in result_json:
                    if isinstance(col, dict):
                        combined_output += f"| {col.get('name', '')} "
                        combined_output += f"| {col.get('type', '')} "
                        combined_output += f"| {col.get('kind', '')} "
                        combined_output += f"| {col.get('null?', '')} "
                        combined_output += f"| {col.get('default', '')} "
                        combined_output += f"| {col.get('primary key', '')} "
                        combined_output += f"| {col.get('unique key', '')} "
                        combined_output += f"| {col.get('check', '')} "
                        combined_output += f"| {col.get('expression', '')} "
                        combined_output += f"| {col.get('comment', '')} "
                        combined_output += f"| {col.get('policy name', '')} "
                        combined_output += f"| {col.get('privacy domain', '')} |\n"
                    else:
                        combined_output += (
                            f"| Column data is not a dictionary: {col} |\n"
                        )

            except Exception as e:
                combined_output += f"\n### {table_name}\nError: {str(e)}\n"
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    import traceback

                    print(
                        f"[DEBUG] Error getting table info for {table_name}: {traceback.format_exc()}"
                    )

        return combined_output.strip(), all_definitions

    elif tool_name == "list_databases":
        try:
            result = await asyncio.to_thread(client.list_databases)
            output = "Available databases:\n" + "\n".join(f"- {db}" for db in result)
            return output, result
        except Exception as e:
            error_msg = f"Error listing databases: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "list_schemas":
        database = tool_args.get("database")
        try:
            result = await asyncio.to_thread(client.list_schemas, database)
            db_name = database or "current database"
            output = f"Schemas in {db_name}:\n" + "\n".join(f"- {s}" for s in result)
            return output, result
        except Exception as e:
            error_msg = f"Error listing schemas: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "list_tables_in":
        database = tool_args.get("database", "")
        schema = tool_args.get("schema", "")
        try:
            result = await asyncio.to_thread(client.list_tables_in, database, schema)
            output = f"Tables in {database}.{schema}:\n" + "\n".join(
                f"- {t}" for t in result
            )
            return output, result
        except Exception as e:
            error_msg = f"Error listing tables: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "validate_query":
        query = tool_args.get("query", "")
        try:
            await asyncio.to_thread(client.validate_query, query)
            output = "Query is valid."
            return output, {"valid": True}
        except Exception as e:
            error_msg = f"Query validation failed: {str(e)}"
            return error_msg, {"valid": False, "error": str(e)}

    elif tool_name == "execute_query":
        query = tool_args.get("query", "")
        try:
            result = await asyncio.to_thread(client.execute_query, query)
            result_json = json.loads(result)
            row_data = result_json.get("rowData", [])
            num_rows = len(row_data)

            if num_rows == 0:
                output = "Query executed successfully. Returned 0 rows."
            else:
                # Get headers from the first row
                headers = list(row_data[0].keys()) if row_data else []

                # Check if output would be too large (>20k chars for safety)
                # Estimate size: header + rows
                estimated_size = (
                    len("| " + " | ".join(headers) + " |\n") * 2
                )  # Header + separator
                for row in row_data[:10]:  # Sample first 10 rows to estimate
                    values = [str(row.get(h, "")) for h in headers]
                    estimated_size += len("| " + " | ".join(values) + " |\n")
                estimated_size = (
                    estimated_size * (num_rows / 10)
                    if num_rows > 10
                    else estimated_size
                )

                if estimated_size > 20000:  # If likely to exceed safe limit
                    # Return data with instructions for chunked display
                    output = f"Query returned {num_rows} rows (ALL DATA FOLLOWS - NO TRUNCATION):\n\n"
                    output += (
                        "IMPORTANT: This is the COMPLETE dataset. Display it ALL.\n\n"
                    )
                    output += "| " + " | ".join(headers) + " |\n"
                    output += "|" + "|".join(["---" for _ in headers]) + "|\n"

                    # Include ALL rows but with a marker for the LLM
                    rows_added = 0
                    current_chunk = output

                    for row in row_data:
                        values = []
                        for h in headers:
                            val = row.get(h, "")
                            if val is None:
                                val_str = "NULL"
                            else:
                                val_str = str(val)
                            values.append(val_str)
                        row_text = "| " + " | ".join(values) + " |\n"

                        # Check if adding this row would exceed chunk size
                        if (
                            len(current_chunk) + len(row_text) > 15000
                            and rows_added > 0
                        ):
                            # Add continuation marker
                            current_chunk += f"\n[CONTINUE WITH ROWS {rows_added+1}-{num_rows} IN NEXT OUTPUT]\n"
                            break

                        current_chunk += row_text
                        rows_added += 1

                    output = current_chunk

                    # If we didn't add all rows, store the rest for continuation
                    if rows_added < num_rows:
                        remaining_rows = row_data[rows_added:]
                        # Store remaining data in cache for continuation
                        continuation_key = f"query_continuation_{tool_call.id}"
                        continuation_data = {
                            "headers": headers,
                            "remaining_rows": remaining_rows,
                            "start_row": rows_added + 1,
                            "total_rows": num_rows,
                        }
                        await asyncio.to_thread(
                            client.set_conversation_data,
                            tool_args.get("conversation_id", "default"),
                            continuation_key,
                            json.dumps(continuation_data),
                        )
                        output += (
                            f"\n\nNOTE: Displaying rows 1-{rows_added} of {num_rows}. "
                        )
                        output += (
                            f"ALL {num_rows} rows are available and MUST be shown. "
                        )
                        output += "Continue displaying the remaining rows immediately."
                else:
                    # Small enough to display normally
                    output = f"Query returned {num_rows} rows (displaying all):\n\n"
                    output += "| " + " | ".join(headers) + " |\n"
                    output += "|" + "|".join(["---" for _ in headers]) + "|\n"

                    for row in row_data:
                        values = []
                        for h in headers:
                            val = row.get(h, "")
                            if val is None:
                                val_str = "NULL"
                            else:
                                val_str = str(val)
                            values.append(val_str)
                        output += "| " + " | ".join(values) + " |\n"

            # Log for debugging
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] execute_query returned ALL {num_rows} rows")
                print(f"[DEBUG] Full output size: {len(output)} characters")

            return output, result_json

        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Query execution error: {e}")
            return error_msg, {"error": str(e)}

    elif tool_name == "continue_output":
        continuation_key = tool_args.get("continuation_key", "")
        conv_id = tool_args.get("conversation_id", "default")

        try:
            continuation_data = await asyncio.to_thread(
                client.get_conversation_data, conv_id, continuation_key
            )

            if not continuation_data:
                return "No continuation data found.", {"error": "No continuation data"}

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

            return output, {"rows_displayed": len(remaining_rows)}

        except Exception as e:
            error_msg = f"Error continuing output: {str(e)}"
            return error_msg, {"error": str(e)}

    else:
        error_msg = f"Unknown tool: {tool_name}"
        return error_msg, {"error": "Unknown tool"}


def get_tool_definitions(client: SnowflakeAI) -> list:
    """Get the list of available tool definitions."""
    return [
        client.get_table_sample_data_tool(),
        client.get_table_definition_tool(),
        client.get_multiple_table_definitions_tool(),
        client.list_databases_tool(),
        client.list_schemas_tool(),
        client.list_tables_in_tool(),
        client.validate_query_tool(),
        client.execute_query_tool(),
    ]
