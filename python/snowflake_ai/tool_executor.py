"""Tool executor for Snowflake AI server."""

import json
import os
import shutil
import subprocess
from typing import Any, Tuple

from ._snowflake_ai import SnowflakeAI, ToolCall


def _find_snow_cli_binary():
    """Find the snow CLI binary."""
    return shutil.which("snow")


async def execute_tool(
    tool_call: ToolCall,
    client: SnowflakeAI,
) -> Tuple[str, Any]:
    """
    Execute a tool call and return formatted output for both LLM and user.

    Returns:
        Tuple of (llm_output, raw_data)
        - llm_output: The data to send to the LLM
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

            return output, result_json

        except Exception as e:
            error_msg = f"Error getting sample data for {table_name}: {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "get_table_schema":
        table_name = tool_args.get("table_name", "")
        try:
            result = await asyncio.to_thread(client.get_table_info, table_name)
            result_json = json.loads(result)

            # Extract columns list from the result dict
            columns = result_json.get("columns", [])
            if not isinstance(columns, list):
                return (
                    f"Error: Invalid format for table schema for {table_name}.",
                    {"error": "Invalid format"},
                )

            output = f"Table schema for {table_name}:\n\n"
            output += "| Column Name | Data Type | Nullable | Default |\n"
            output += "|---|---|---|---|\n"

            for col in columns:
                output += f"| {col.get('COLUMN_NAME', '')} "
                output += f"| {col.get('DATA_TYPE', '')} "
                output += f"| {col.get('IS_NULLABLE', '')} "
                output += f"| {col.get('COLUMN_DEFAULT', '')} |\n"

            return output, result_json

        except Exception as e:
            error_msg = f"Error getting table schema for {table_name}: {str(e)}"
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

                # Limit output to 50 rows for LLM context
                limit = 50
                display_rows = min(num_rows, limit)

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

    elif tool_name == "execute_statement":
        statement = (tool_args.get("statement", "") or "").strip()
        if not statement:
            return "Error: 'statement' argument is required.", {
                "error": "Missing statement"
            }

        try:
            result = await asyncio.to_thread(client.execute_statement, statement)
            try:
                rows = json.loads(result)
            except json.JSONDecodeError:
                return (
                    "Statement executed successfully. Raw response returned.",
                    {"raw": result},
                )

            if not isinstance(rows, list):
                return (
                    "Statement executed successfully.",
                    rows,
                )

            if not rows:
                return "Statement executed successfully. No rows returned.", []

            headers = []
            for row in rows:
                if isinstance(row, dict):
                    for key in row.keys():
                        if key not in headers:
                            headers.append(key)

            output = f"Statement executed successfully. Returned {len(rows)} rows.\n\n"
            if headers:
                output += "| " + " | ".join(headers) + " |\n"
                output += "|" + "|".join(["---" for _ in headers]) + "|\n"

                for row in rows:
                    if isinstance(row, dict):
                        values = []
                        for header in headers:
                            val = row.get(header, "")
                            if val is None:
                                values.append("NULL")
                            else:
                                values.append(str(val))
                        output += "| " + " | ".join(values) + " |\n"

            return output.strip(), rows

        except Exception as e:
            error_msg = f"Statement execution failed: {str(e)}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(f"[DEBUG] Statement execution error: {e}")
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

    elif tool_name in ("extract_answer", "sentiment", "summarize", "translate"):
        snow_cli = _find_snow_cli_binary()
        if not snow_cli:
            return "Snowflake CLI not found.", {"error": "Snowflake CLI not found"}

        command = [snow_cli, "cortex", tool_name.replace("_", "-")]
        if tool_name == "extract_answer":
            command.extend(
                [
                    "--question",
                    tool_args["question"],
                    "--document",
                    tool_args["document"],
                ]
            )
        elif tool_name == "translate":
            command.extend(["--text", tool_args["text"], "--to", tool_args["to_lang"]])
            if "from_lang" in tool_args:
                command.extend(["--from", tool_args["from_lang"]])
        else:  # sentiment, summarize
            command.append(tool_args["text"])

        try:
            result = await asyncio.to_thread(
                subprocess.run, command, capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout)
            return result.stdout, data
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error executing {tool_name}: {e}"
            return error_msg, {"error": str(e)}

    elif tool_name == "read_document":
        file_name = tool_args.get("file_name", "")
        page_numbers = tool_args.get("page_numbers", [])

        if not file_name:
            return "Error: file_name is required", {"error": "file_name required"}

        try:
            # Get user schema
            snowflake_user = await asyncio.to_thread(client.get_current_user)
            sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
            user_schema = f"USER_{sanitized_user}".upper()

            # Build query
            if page_numbers:
                page_list = ",".join(str(p) for p in page_numbers)
                query = f"""
                SELECT PAGE_NUMBER, PAGE_CONTENT 
                FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS 
                WHERE FILE_NAME = '{file_name}' 
                AND PAGE_NUMBER IN ({page_list})
                ORDER BY PAGE_NUMBER
                """
            else:
                query = f"""
                SELECT PAGE_NUMBER, PAGE_CONTENT 
                FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS 
                WHERE FILE_NAME = '{file_name}'
                ORDER BY PAGE_NUMBER
                """

            result = await asyncio.to_thread(client.execute_query, query)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if not row_data:
                return (
                    f"No content found for document '{file_name}'. The document may not be parsed yet.",
                    {"error": "No content"},
                )

            # Format document content
            output = f"**Document: {file_name}** ({len(row_data)} pages)\n\n"
            for row in row_data:
                page_num = row.get("PAGE_NUMBER", row.get("page_number", "?"))
                content = row.get("PAGE_CONTENT", row.get("page_content", ""))
                output += f"---\n**Page {page_num}:**\n{content}\n\n"

            return output, {
                "file_name": file_name,
                "pages": len(row_data),
                "content": row_data,
            }

        except Exception as e:
            error_msg = f"Error reading document '{file_name}': {str(e)}"
            return error_msg, {"error": str(e)}

    elif tool_name == "remove_document":
        file_name = tool_args.get("file_name", "")

        if not file_name:
            return "Error: file_name is required", {"error": "file_name required"}

        try:
            from .helpers import remove_file_from_stage

            # Add .gz extension if not present (files are gzipped on upload)
            file_to_remove = file_name
            if not file_to_remove.endswith(".gz"):
                file_to_remove = f"{file_name}.gz"

            success, message = await remove_file_from_stage(client, file_to_remove)

            if success:
                return (
                    f"Successfully removed document '{file_name}' and any associated parsed content.",
                    {"success": True, "file_name": file_name, "message": message},
                )
            else:
                return (
                    f"Failed to remove document '{file_name}': {message}",
                    {"success": False, "error": message},
                )

        except Exception as e:
            error_msg = f"Error removing document '{file_name}': {str(e)}"
            return error_msg, {"error": str(e)}


def get_tool_definitions(client: SnowflakeAI) -> list:
    """Get the list of available tool definitions."""
    raw_tools = [
        client.get_table_sample_data_tool(),
        client.get_table_schema_tool(),
        client.get_multiple_table_definitions_tool(),
        client.list_databases_tool(),
        client.list_schemas_tool(),
        client.list_tables_in_tool(),
        client.validate_query_tool(),
        client.execute_query_tool(),
        client.execute_statement_tool(),
        client.extract_answer_tool(),
        client.sentiment_tool(),
        client.summarize_tool(),
        client.translate_tool(),
    ]

    # Add get_widget_data tool for fetching dashboard widget data
    get_widget_data_tool = {
        "type": "function",
        "function": {
            "name": "get_widget_data",
            "description": "Fetch data from dashboard widgets. Use this tool when the user asks about charts, tables, or other visualizations on their dashboard. The data will be retrieved from the widget and made available for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "widget_uuids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of widget UUIDs to fetch data from. If not specified, data from all available widgets will be fetched.",
                    }
                },
                "required": [],
            },
        },
    }
    raw_tools.append(get_widget_data_tool)

    # Add read_document tool for easy document content retrieval
    read_document_tool = {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the content of an uploaded document from Snowflake. Use this when the user asks about or references a document they've uploaded. The document must have been parsed (check 'parsed' status in available documents list).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The exact filename of the document to read (e.g., 'technology-investment.pdf'). Match user's description to the available documents list.",
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional list of specific page numbers to retrieve. If not specified, all pages are returned.",
                    },
                },
                "required": ["file_name"],
            },
        },
    }
    raw_tools.append(read_document_tool)

    # Add remove_document tool for deleting uploaded documents
    remove_document_tool = {
        "type": "function",
        "function": {
            "name": "remove_document",
            "description": "Remove an uploaded document from Snowflake storage. Use this when the user wants to delete a document they've uploaded. This will remove both the file from the stage and any parsed content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The filename to remove (e.g., 'technology-investment.pdf' or 'widget_pdf_abc123.pdf.gz'). Can include or exclude the .gz extension.",
                    },
                },
                "required": ["file_name"],
            },
        },
    }
    raw_tools.append(remove_document_tool)

    normalized_tools: list[Any] = []
    for tool in raw_tools:
        if isinstance(tool, str):
            try:
                parsed = json.loads(tool)
                normalized_tools.append(parsed)
            except json.JSONDecodeError:
                # Fall back to simple wrapper so downstream code can inspect name
                normalized_tools.append({"function": {"name": tool}})
        else:
            normalized_tools.append(tool)

    return normalized_tools
