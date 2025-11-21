"""Tool execution logic for Snowflake AI."""

import json
from typing import Any

from ._snowflake_ai import SnowflakeAI, ToolCall


async def run_in_thread(func, *args, **kwargs):
    """Run a function in a background thread."""
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def summarize_message_content(content: str, role: str, max_length: int = 500) -> str:
    """
    Intelligently summarize message content instead of truncating.

    Args:
        content: The message content to summarize
        role: The role of the message (human, assistant, tool, system)
        max_length: Maximum length before summarization is needed

    Returns:
        Original content if short, or a meaningful summary if long
    """
    if len(content) <= max_length:
        return content

    # For different message types, extract different key information
    if role == "tool":
        # For tool outputs, preserve structure and key data
        lines = content.split("\n")

        # Check if it's a table (markdown or data)
        if "|" in content or "rows:" in content.lower():
            # Extract table metadata
            row_count = content.count("\n")
            if "rows:" in content.lower():
                # Try to extract row count from text
                import re

                match = re.search(r"(\d+)\s+rows", content.lower())
                if match:
                    row_count = match.group(1)

            # Get first few lines and last line
            preview_lines = lines[:5]
            summary = "\n".join(preview_lines)
            summary += f"\n... (table with ~{row_count} rows, showing first few) ..."
            return summary

        # For schema definitions or structured data
        elif "schema" in content.lower() or "column" in content.lower():
            preview_lines = lines[:10]
            summary = "\n".join(preview_lines)
            summary += (
                f"\n... (schema definition continues, {len(lines)} total lines) ..."
            )
            return summary

        # For list outputs
        elif content.strip().startswith("-") or content.strip().startswith("•"):
            items = [line for line in lines if line.strip().startswith(("-", "•"))]
            if len(items) > 10:
                preview = "\n".join(items[:10])
                summary = f"{preview}\n... (and {len(items) - 10} more items)"
                return summary

    elif role == "assistant":
        # For assistant responses, keep introduction and conclusion
        lines = content.split("\n")
        if len(lines) > 15:
            intro = "\n".join(lines[:7])
            conclusion = "\n".join(lines[-3:])
            summary = (
                f"{intro}\n\n... (detailed explanation omitted) ...\n\n{conclusion}"
            )
            return summary

    elif role == "human":
        # For human messages, try to extract the core question/request
        # Keep first 300 chars which usually contain the main question
        summary = content[:300]
        if len(content) > 300:
            summary += "... (additional context provided)"
        return summary

    # Default: intelligent truncation with context preservation
    # Take beginning and end
    start = content[:250]
    end = content[-200:]
    return f"{start}\n\n... ({len(content)} chars total, middle section summarized) ...\n\n{end}"


async def execute_tool(
    tool_call: ToolCall,
    client: SnowflakeAI,
) -> tuple[str, dict[str, Any] | None]:
    """
    Execute a single tool call.

    Args:
        tool_call: The tool call to execute
        client: SnowflakeAI client instance

    Returns:
        Tuple of (output_for_llm, details_for_cache)

    Raises:
        ValueError: If tool name is unknown or arguments are invalid
    """
    tool_name = tool_call.function.name
    tool_args_str = tool_call.function.arguments

    try:
        tool_args = json.loads(tool_args_str)
    except json.JSONDecodeError as e:
        err_msg = "Error: Invalid JSON in tool arguments"
        return err_msg, {"error": str(e)}

    current_tool_output_for_llm = None
    details_for_cache = None

    try:
        if tool_name == "list_databases":
            databases = await run_in_thread(client.list_databases)
            summary_for_llm = (
                f"Found {len(databases)} databases: {', '.join(databases)}"
            )
            current_tool_output_for_llm = summary_for_llm

        elif tool_name == "get_table_sample_data":
            table_name = tool_args.get("table_name")
            if not table_name:
                raise ValueError("table_name not provided")
            result_json_str = await run_in_thread(
                client.get_table_sample_data_rust, table_name
            )
            result_json = json.loads(result_json_str)

            # Convert to markdown table
            columns = [col["field"] for col in result_json.get("columnDefs", [])]
            rows = result_json.get("rowData", [])

            markdown_table = f"Sample data for {table_name}:\n\n"
            markdown_table += "| " + " | ".join(columns) + " |\n"
            markdown_table += "|" + "---|" * len(columns) + "\n"

            for row in rows:
                values = [str(row.get(col, "")) for col in columns]
                markdown_table += "| " + " | ".join(values) + " |\n"

            current_tool_output_for_llm = markdown_table

        elif tool_name == "get_table_definition":
            table_name_fqn = tool_args.get("table_name")
            if not table_name_fqn:
                raise ValueError("table_name not provided")
            result_json_str = await run_in_thread(client.get_table_info, table_name_fqn)
            result_data = json.loads(result_json_str)
            columns = result_data.get("columns", [])
            primary_keys = result_data.get("primary_keys", [])

            pk_columns = set()
            for pk in primary_keys:
                if isinstance(pk, dict) and "column_name" in pk:
                    pk_columns.add(pk["column_name"])

            markdown_table = f"Schema for {table_name_fqn}:\n\n"
            markdown_table += "| name | type | kind | null? | default | primary key | unique key | check | expression | comment | policy name | privacy domain |\n"
            markdown_table += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"

            for col in columns:
                name = col.get("name", "")
                dtype = col.get("type", "")
                kind = col.get("kind", "")
                nullable = col.get("null?", "")
                default = col.get("default", "null")
                pk = "Y" if name in pk_columns else "N"
                uk = col.get("unique key", "N")
                check = col.get("check", "null")
                expr = col.get("expression", "null")
                comment = col.get("comment", "null")
                policy = col.get("policy name", "null")
                privacy = col.get("privacy domain", "null")
                markdown_table += (
                    f"| {name} | {dtype} | {kind} | {nullable} | {default} | {pk} | "
                    f"{uk} | {check} | {expr} | {comment} | {policy} | {privacy} |\n"
                )

            current_tool_output_for_llm = markdown_table

        elif tool_name == "get_multiple_table_definitions":
            table_names = tool_args.get("table_names", [])
            all_markdown_tables = []

            for table_name in table_names:
                try:
                    result_json_str = await run_in_thread(
                        client.get_table_info, table_name
                    )
                    result_data = json.loads(result_json_str)
                    columns = result_data.get("columns", [])
                    primary_keys = result_data.get("primary_keys", [])

                    pk_columns = set()
                    for pk in primary_keys:
                        if isinstance(pk, dict) and "column_name" in pk:
                            pk_columns.add(pk["column_name"])

                    markdown_table = f"Schema for {table_name}:\n\n"
                    markdown_table += "| name | type | kind | null? | default | primary key | unique key | check | expression | comment | policy name | privacy domain |\n"
                    markdown_table += (
                        "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
                    )

                    for col in columns:
                        name = col.get("name", "")
                        dtype = col.get("type", "")
                        kind = col.get("kind", "")
                        nullable = col.get("null?", "")
                        default = col.get("default", "null")
                        pk = "Y" if name in pk_columns else "N"
                        uk = col.get("unique key", "N")
                        check = col.get("check", "null")
                        expr = col.get("expression", "null")
                        comment = col.get("comment", "null")
                        policy = col.get("policy name", "null")
                        privacy = col.get("privacy domain", "null")
                        markdown_table += (
                            f"| {name} | {dtype} | {kind} | {nullable} | {default} | {pk} | "
                            f"{uk} | {check} | {expr} | {comment} | {policy} | {privacy} |\n"
                        )

                    all_markdown_tables.append(markdown_table)
                except Exception as e:
                    all_markdown_tables.append(
                        f"Error getting schema for {table_name}: {str(e)}"
                    )

            summary_for_llm = "\n\n".join(all_markdown_tables)
            current_tool_output_for_llm = summary_for_llm

        elif tool_name == "list_schemas":
            database = tool_args.get("database")
            schemas = await run_in_thread(client.list_schemas, database)
            db_name = database or "the current database"
            summary_for_llm = (
                f"Found {len(schemas)} schemas in {db_name}: {', '.join(schemas)}"
            )
            current_tool_output_for_llm = summary_for_llm

        elif tool_name == "list_tables_in":
            database = tool_args.get("database")
            schema = tool_args.get("schema")
            if not database or not schema:
                raise ValueError("database and schema are required")
            tables = await run_in_thread(client.list_tables_in, database, schema)
            summary_for_llm = f"Found {len(tables)} tables in {database}.{schema}: {', '.join(tables)}"
            current_tool_output_for_llm = summary_for_llm

        elif tool_name == "validate_query":
            query = tool_args.get("query")
            if not query:
                raise ValueError("query not provided")
            try:
                is_valid = await run_in_thread(client.validate_query, query)
                summary_for_llm = (
                    f"Query validation: {'Valid ✓' if is_valid else 'Invalid ✗'}"
                )
                current_tool_output_for_llm = summary_for_llm
            except Exception as validation_error:
                current_tool_output_for_llm = (
                    f"Query validation failed: {str(validation_error)}"
                )

        elif tool_name == "execute_query":
            query = tool_args.get("query")
            if not query:
                raise ValueError("query not provided")
            result_json_str = await run_in_thread(client.execute_query, query)
            result_data = json.loads(result_json_str)
            row_count = result_data.get("rowCount", 0)

            if row_count > 0:
                # Format as markdown table
                columns = result_data.get("columns", [])
                rows = result_data.get("rows", [])

                markdown_table = "| " + " | ".join(columns) + " |\n"
                markdown_table += "|" + "---|" * len(columns) + "\n"

                for row in rows:
                    markdown_table += "| " + " | ".join(str(v) for v in row) + " |\n"

                current_tool_output_for_llm = f"Query executed successfully. Returned {row_count} rows:\n\n{markdown_table}"
            else:
                current_tool_output_for_llm = (
                    "Query executed successfully. No rows returned."
                )

        # Note: get_conversation_history tool is not exposed via tool calling
        # because the Rust client only accepts Tool objects from its own methods.
        # Conversation history is accessible via /list_conv_data slash command instead.

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    except Exception as e:
        err_msg = f"Tool execution failed: {str(e)}"
        current_tool_output_for_llm = err_msg
        details_for_cache = {"error": str(e)}

    return current_tool_output_for_llm, details_for_cache


def get_tool_definitions(client: SnowflakeAI) -> list:
    """
    Get all available tool definitions from the client.

    Args:
        client: SnowflakeAI client instance

    Returns:
        List of tool definitions (Tool objects from Rust bindings)
    """
    # Only return tools that are properly defined in the Rust client
    # All these return Tool objects that can be serialized properly
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
