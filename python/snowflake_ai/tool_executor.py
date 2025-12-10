"""Tool execution for Snowflake AI."""

import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import traceback
from datetime import datetime, timezone
from typing import Any

from openbb_ai import reasoning_step

from ._snowflake_ai import SnowflakeAI, ToolCall
from .document_processor import DocumentProcessor
from .helpers import to_sse
from .logger import get_logger

logger = get_logger(__name__)

SQL_CACHE_KEY = "text2sql_cache_v1"
SQL_CACHE_MAX_ENTRIES = 5
SQL_CACHE_ROW_LIMIT = 500
SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join|into|update|table)\s+([\w\.\"$]+)",
    re.IGNORECASE,
)


def _normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql or "").strip()


def _hash_sql(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


async def _load_sql_cache(client: SnowflakeAI, conv_id: str) -> dict:
    try:
        raw = await asyncio.to_thread(
            client.get_conversation_data, conv_id, SQL_CACHE_KEY
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to load SQL cache: %s", exc)
        return {}

    if not raw:
        return {}

    try:
        cache = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return cache if isinstance(cache, dict) else {}


async def _persist_sql_cache(client: SnowflakeAI, conv_id: str, cache: dict):
    try:
        await asyncio.to_thread(
            client.set_conversation_data,
            conv_id,
            SQL_CACHE_KEY,
            json.dumps(cache),
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to persist SQL cache: %s", exc)


def _shrink_cache(cache: dict):
    if len(cache) <= SQL_CACHE_MAX_ENTRIES:
        return

    def _entry_ts(item):
        value = item[1].get("executed_at")
        if isinstance(value, str):
            try:
                ts = value.replace("Z", "+00:00")
                return datetime.fromisoformat(ts)
            except ValueError:
                return datetime.min.replace(tzinfo=timezone.utc)
        return datetime.min.replace(tzinfo=timezone.utc)

    sorted_items = sorted(cache.items(), key=_entry_ts, reverse=True)
    trimmed = dict(sorted_items[:SQL_CACHE_MAX_ENTRIES])
    cache.clear()
    cache.update(trimmed)


def _split_identifier(identifier: str) -> list[str]:
    parts: list[str] = []
    current = []
    in_quotes = False
    for char in identifier:
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "." and not in_quotes:
            if current:
                parts.append("".join(current))
                current = []
            continue
        if not in_quotes and char in ",;()":
            break
        current.append(char)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


async def _get_context_defaults(client: SnowflakeAI) -> tuple[str | None, str | None]:
    try:
        database = await asyncio.to_thread(client.get_current_database)
    except Exception:  # pragma: no cover - defensive
        database = None
    try:
        schema = await asyncio.to_thread(client.get_current_schema)
    except Exception:  # pragma: no cover - defensive
        schema = None
    return database, schema


def _extract_tables_from_sql(
    sql: str, default_db: str | None, default_schema: str | None
) -> list[dict[str, str]]:
    tables: list[dict[str, str]] = []
    seen = set()
    for match in SQL_TABLE_PATTERN.finditer(sql):
        identifier = match.group(1)
        parts = _split_identifier(identifier)
        if not parts:
            continue
        if len(parts) == 3:
            database, schema, table = parts
        elif len(parts) == 2:
            database = default_db
            schema, table = parts
        else:
            database = default_db
            schema = default_schema
            table = parts[0]

        key = (database or "", schema or "", table)
        if key in seen or not table:
            continue
        seen.add(key)
        tables.append(
            {
                "database": (database or "").strip('"'),
                "schema": (schema or "").strip('"'),
                "table": table.strip('"'),
            }
        )
    return tables


async def _capture_table_freshness(
    client: SnowflakeAI, tables: list[dict[str, str]]
) -> list[dict[str, str]]:
    freshness = []
    for table in tables:
        db = table.get("database")
        schema = table.get("schema")
        name = table.get("table")
        if not (db and schema and name):
            continue
        try:
            last_altered = await asyncio.to_thread(
                client.get_table_last_altered,
                db,
                schema,
                name,
            )
        except Exception:  # pragma: no cover - defensive
            last_altered = None
        if last_altered:
            freshness.append({**table, "last_altered": last_altered})
    return freshness


async def _is_cache_entry_stale(client: SnowflakeAI, entry: dict) -> bool:
    tables = entry.get("tables") or []
    if not tables:
        return True

    for table in tables:
        db = table.get("database")
        schema = table.get("schema")
        name = table.get("table")
        cached_ts = table.get("last_altered")
        if not (db and schema and name and cached_ts):
            return True
        try:
            current = await asyncio.to_thread(
                client.get_table_last_altered,
                db,
                schema,
                name,
            )
        except Exception:  # pragma: no cover - defensive
            return True
        if not current:
            return True
        if current > cached_ts:
            return True
    return False


async def _cache_query_result(
    client: SnowflakeAI,
    conv_id: str,
    cache: dict,
    query_hash: str,
    query: str,
    result_json: dict,
    row_count: int,
    tables: list[dict[str, str]],
):
    if not tables:
        return

    freshness = await _capture_table_freshness(client, tables)
    if not freshness:
        return

    cached_copy = json.loads(json.dumps(result_json))
    if isinstance(cached_copy.get("rowData"), list):
        cached_copy["rowData"] = cached_copy["rowData"][:SQL_CACHE_ROW_LIMIT]

    cache[query_hash] = {
        "query": query,
        "executed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "row_count": row_count,
        "tables": freshness,
        "result": cached_copy,
    }
    _shrink_cache(cache)
    await _persist_sql_cache(client, conv_id, cache)


def _find_snow_cli_binary():
    """Find the snow CLI binary."""
    return shutil.which("snow")


def _extract_tables_from_text(text: str) -> list[dict]:
    """Extract markdown-style tables from text content.

    Parses text looking for pipe-delimited table syntax and extracts
    the actual table structure with headers and rows.

    Parameters
    ----------
    text : str
        The text content to parse for tables

    Returns
    -------
    list[dict]
        List of table dicts with keys:
        - 'headers': list of column headers
        - 'rows': list of row data (each row is a list of cell values)
        - 'raw_text': the original table text
    """
    tables = []
    lines = text.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for lines with pipe characters (potential table row)
        if "|" in line and line.count("|") >= 2:
            table_lines = []
            table_start = i

            # Collect consecutive lines that look like table rows
            while i < len(lines):
                current_line = lines[i].strip()
                # Stop if line doesn't have pipes or is empty
                if not current_line or ("|" not in current_line and current_line):
                    # Allow one blank line within table, but not more
                    if i + 1 < len(lines) and "|" in lines[i + 1]:
                        i += 1
                        continue
                    break
                table_lines.append(current_line)
                i += 1

            # Need at least 2 lines for a valid table (header + data or separator)
            if len(table_lines) >= 2:
                headers = []
                rows = []
                separator_idx = -1

                # Find the separator line (contains only |, -, :, spaces)
                for idx, tline in enumerate(table_lines):
                    # Check if this is a separator line
                    cleaned = (
                        tline.replace("|", "")
                        .replace("-", "")
                        .replace(":", "")
                        .replace(" ", "")
                    )
                    if not cleaned or all(c in "-:|" for c in tline.replace(" ", "")):
                        separator_idx = idx
                        break

                # Parse headers (line before separator, or first line)
                header_idx = separator_idx - 1 if separator_idx > 0 else 0
                if header_idx >= 0 and header_idx < len(table_lines):
                    header_line = table_lines[header_idx]
                    # Split by | and clean up
                    parts = [p.strip() for p in header_line.split("|")]
                    # Remove empty parts from start/end (from leading/trailing |)
                    headers = [p for p in parts if p]

                # Parse data rows (lines after separator)
                data_start = separator_idx + 1 if separator_idx >= 0 else 1
                for row_line in table_lines[data_start:]:
                    # Skip separator-like lines
                    cleaned = (
                        row_line.replace("|", "")
                        .replace("-", "")
                        .replace(":", "")
                        .replace(" ", "")
                    )
                    if not cleaned:
                        continue

                    parts = [p.strip() for p in row_line.split("|")]
                    row_data = [
                        p for p in parts if p or len(parts) > 2
                    ]  # Keep empty cells in middle
                    # Only keep if it has reasonable cell count
                    if row_data and (not headers or len(row_data) >= len(headers) - 1):
                        rows.append(row_data[: len(headers)] if headers else row_data)

                # Only add if we have meaningful content
                if headers or rows:
                    tables.append(
                        {
                            "headers": headers,
                            "rows": rows,
                            "raw_text": "\n".join(table_lines),
                        }
                    )
        else:
            i += 1

    return tables


async def execute_tool(
    tool_call: ToolCall,
    client: SnowflakeAI,
    conv_id: str = "default",
):
    """
    Execute a tool call and yield reasoning steps and final output.

    Parameters
    ----------
    tool_call : ToolCall
        The tool call to execute
    client : SnowflakeAI
        The Snowflake client
    conv_id : str
        Conversation ID for accessing cached metadata

    Yields:
        Reasoning steps and a final Tuple of (llm_output, raw_data)
    """
    tool_name = tool_call.function.name
    tool_args_str = tool_call.function.arguments

    try:
        tool_args = json.loads(tool_args_str) if tool_args_str else {}
    except json.JSONDecodeError:
        tool_args = {"raw": tool_args_str}

    # Inject conversation ID for tools that need cached metadata
    tool_args["_conversation_id"] = conv_id

    # Some tools provide their own reasoning steps, so skip the generic one
    if tool_name not in ["text2sql"]:
        yield to_sse(
            reasoning_step(
                f"Executing tool: {tool_name} with arguments: {tool_args_str}",
                event_type="INFO",
            )
        )

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

            yield output, result_json
            return

        except Exception as e:
            error_msg = f"Error getting sample data for {table_name}: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "get_table_schema":
        table_name = tool_args.get("table_name", "")
        try:
            result = await asyncio.to_thread(client.get_table_info, table_name)
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

            yield (output, result_json)
            return

        except Exception as e:
            error_msg = f"Error getting table schema for {table_name}: {str(e)}"
            yield (error_msg, {"error": str(e)})
            return

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
                    logger.error(
                        "Error getting table info for %s: %s",
                        table_name,
                        traceback.format_exc(),
                    )

        yield combined_output.strip(), all_definitions
        return

    elif tool_name == "list_databases":
        try:
            result = await asyncio.to_thread(client.list_databases)
            output = "Available databases:\n" + "\n".join(f"- {db}" for db in result)
            yield output, result
            return
        except Exception as e:
            error_msg = f"Error listing databases: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "list_schemas":
        database = tool_args.get("database")
        try:
            result = await asyncio.to_thread(client.list_schemas, database)
            db_name = database or "current database"
            output = f"Schemas in {db_name}:\n" + "\n".join(f"- {s}" for s in result)
            yield output, result
            return
        except Exception as e:
            error_msg = f"Error listing schemas: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "list_tables_in":
        database = tool_args.get("database", "")
        schema = tool_args.get("schema", "")
        try:
            result = await asyncio.to_thread(client.list_tables_in, database, schema)
            output = f"Tables in {database}.{schema}:\n" + "\n".join(
                f"- {t}" for t in result
            )
            yield output, result
            return
        except Exception as e:
            error_msg = f"Error listing tables: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "text2sql":
        prompt = tool_args.get("prompt") or tool_args.get("question")
        if not prompt:
            yield "Error: 'prompt' argument is required for text2sql.", {
                "error": "Missing prompt"
            }
            return

        # Set conversation context (database/schema) if stored in settings
        # This allows text2sql to use the correct schema from /use_database, /use_schema commands
        try:
            settings_json = await asyncio.to_thread(
                client.get_or_create_conversation, conv_id, json.dumps({})
            )
            if settings_json:
                settings = json.loads(settings_json)
                db = settings.get("database")
                sch = settings.get("schema")
                if db or sch:
                    await asyncio.to_thread(client.use_conversation_context, db, sch)
        except Exception:
            pass  # Ignore errors, use default database/schema

        # Check if semantic model needs to be generated
        # This is indicated by not having a cached model
        has_cached_model = await asyncio.to_thread(client.has_semantic_model_cache)

        if not has_cached_model:
            yield to_sse(
                reasoning_step(
                    "Preparing semantic model and generating SQL (first run may take 30-60 seconds)...",
                    event_type="INFO",
                )
            )
        else:
            yield to_sse(
                reasoning_step(
                    "Generating SQL using cached semantic model...",
                    event_type="INFO",
                )
            )

        try:
            response = await asyncio.to_thread(client.text2sql, prompt)
            parsed = json.loads(response)
        except Exception as exc:  # pragma: no cover - defensive
            error_msg = f"text2sql generation failed: {exc}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error(error_msg)
            yield error_msg, {"error": str(exc)}
            return

        sql_text = (parsed.get("sql") or "").strip()
        explanation = (parsed.get("explanation") or "").strip()
        request_id = parsed.get("request_id")

        output_sections: list[str] = []
        if explanation:
            output_sections.append(explanation)
        if sql_text:
            output_sections.append(f"```sql\n{sql_text}\n```")
        else:
            output_sections.append("No SQL was generated.")
        if request_id:
            output_sections.append(f"(request_id: {request_id})")

        yield "\n\n".join(output_sections), parsed
        return

    elif tool_name == "validate_query":
        query = tool_args.get("query", "")
        try:
            await asyncio.to_thread(client.validate_query, query)
            output = "Query is valid."
            yield output, {"valid": True}
            return
        except Exception as e:
            error_msg = f"Query validation failed: {str(e)}"
            yield error_msg, {"valid": False, "error": str(e)}
            return

    elif tool_name == "execute_query":
        query = tool_args.get("query", "")
        conv_id = tool_args.get("_conversation_id", "default")
        force_refresh = bool(tool_args.get("force_refresh"))

        normalized_query = _normalize_sql(query)
        query_hash = _hash_sql(normalized_query)
        cache = await _load_sql_cache(client, conv_id)

        if not force_refresh and query_hash in cache:
            cache_entry = cache[query_hash]
            try:
                is_stale = await _is_cache_entry_stale(client, cache_entry)
            except Exception as freshness_error:  # pragma: no cover - defensive
                logger.debug("Cache freshness check failed: %s", freshness_error)
                is_stale = True

            if not is_stale:
                cached_result = cache_entry.get("result")
                if isinstance(cached_result, dict):
                    reuse_msg = cache_entry.get("executed_at", "cached result")
                    yield (
                        f"Reusing cached query result from {reuse_msg} (no schema changes detected).",
                        cached_result,
                    )
                    return

        try:
            result = await asyncio.to_thread(client.execute_query, query)
            result_json = json.loads(result)
            row_data = result_json.get("rowData", [])
            num_rows = len(row_data)

            # Generate table artifact if we have data
            if num_rows > 0 and row_data:
                try:
                    from openbb_ai.helpers import table

                    # Create table artifact with query results
                    table_artifact = table(
                        data=row_data,
                        name="Query Results",
                        description=(
                            query[:200] if len(query) <= 200 else query[:197] + "..."
                        ),
                    )

                    # Yield table artifact for UI
                    yield to_sse(table_artifact)

                except Exception as artifact_error:
                    logger.warning(f"Failed to create table artifact: {artifact_error}")

            # Generate text summary for LLM
            if num_rows == 0:
                output = "Query executed successfully. Returned 0 rows."
            else:
                headers = list(row_data[0].keys()) if row_data else []
                output = (
                    f"Query returned {num_rows} rows with columns: {', '.join(headers)}"
                )

            yield output, result_json

            try:
                default_db, default_schema = await _get_context_defaults(client)
                tables = _extract_tables_from_sql(
                    normalized_query, default_db, default_schema
                )
                await _cache_query_result(
                    client,
                    conv_id,
                    cache,
                    query_hash,
                    normalized_query,
                    result_json,
                    num_rows,
                    tables,
                )
            except Exception as cache_exc:  # pragma: no cover - defensive
                logger.debug("Skipping query cache storage: %s", cache_exc)
            return

        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Query execution error: %s", e)
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "execute_statement":
        statement = (tool_args.get("statement", "") or "").strip()
        if not statement:
            yield "Error: 'statement' argument is required.", {
                "error": "Missing statement"
            }
            return

        try:
            result = await asyncio.to_thread(client.execute_statement, statement)
            try:
                rows = json.loads(result)
            except json.JSONDecodeError:
                yield (
                    "Statement executed successfully. Raw response returned.",
                    {"raw": result},
                )
                return

            if not isinstance(rows, list):
                yield (
                    "Statement executed successfully.",
                    rows,
                )
                return

            if not rows:
                yield "Statement executed successfully. No rows returned.", []
                return

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

            yield output.strip(), rows
            return

        except Exception as e:
            error_msg = f"Statement execution failed: {str(e)}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Statement execution error: %s", e)
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "continue_output":
        continuation_key = tool_args.get("continuation_key", "")
        conv_id = tool_args.get("conversation_id", "default")

        try:
            continuation_data = await asyncio.to_thread(
                client.get_conversation_data, conv_id, continuation_key
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
            return

        except Exception as e:
            error_msg = f"Error continuing output: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "extract_answer":
        # Use AI_EXTRACT SQL function instead of deprecated CLI command
        file_path = tool_args.get("file_path", "")
        questions = tool_args.get("questions", [])

        if not file_path or not questions:
            yield "Error: file_path and questions are required", {
                "error": "Missing required arguments"
            }
            return

        try:
            # Build response format as array of questions
            response_format = json.dumps(questions)

            # Construct AI_EXTRACT query
            query = f"""
            SELECT AI_EXTRACT(
                file => TO_FILE('{file_path}'),
                responseFormat => PARSE_JSON('{response_format}')
            ) as extraction_result
            """

            result = await asyncio.to_thread(client.execute_query, query)
            result_json = json.loads(result)

            row_data = result_json.get("rowData", [])
            if not row_data:
                yield "No extraction results returned", {"error": "No results"}
                return

            extraction = row_data[0].get("EXTRACTION_RESULT", {})

            # Format output
            output = "Extraction results:\n\n"
            if isinstance(extraction, dict):
                response_data = extraction.get("response", {})
                error = extraction.get("error")

                if error:
                    output += f"Error: {error}\n"
                else:
                    for i, question in enumerate(questions):
                        answer = response_data.get(str(i), "No answer found")
                        output += f"**Q: {question}**\n{answer}\n\n"
            else:
                output += str(extraction)

            yield output, extraction
            return

        except Exception as e:
            error_msg = f"Error executing AI_EXTRACT: {e}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name in ("sentiment", "summarize", "translate"):
        snow_cli = _find_snow_cli_binary()
        if not snow_cli:
            yield "Snowflake CLI not found.", {"error": "Snowflake CLI not found"}
            return

        command = [snow_cli, "cortex", tool_name.replace("_", "-")]
        if tool_name == "translate":
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
            yield result.stdout, data
            return
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            error_msg = f"Error executing {tool_name}: {e}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "read_document":
        file_name = tool_args.get("file_name", "")
        page_numbers = tool_args.get("page_numbers", [])

        if not file_name:
            yield "Error: file_name is required", {"error": "file_name required"}
            return

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
                yield (
                    f"No content found for document '{file_name}'. The document may not be parsed yet.",
                    {"error": "No content"},
                )
                return

            # Format document content
            output = f"**Document: {file_name}** ({len(row_data)} pages)\n\n"
            for row in row_data:
                page_num = row.get("PAGE_NUMBER", row.get("page_number", "?"))
                content = row.get("PAGE_CONTENT", row.get("page_content", ""))
                output += f"---\n**Page {page_num}:**\n{content}\n\n"

            yield output, {
                "file_name": file_name,
                "pages": len(row_data),
                "content": row_data,
            }
            return

        except Exception as e:
            error_msg = f"Error reading document '{file_name}': {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "remove_document":
        file_name = tool_args.get("file_name", "")

        if not file_name:
            yield "Error: file_name is required", {"error": "file_name required"}
            return

        try:
            doc_proc = DocumentProcessor.instance()

            # Add .gz extension if not present (files are gzipped on upload)
            file_to_remove = file_name
            if not file_to_remove.endswith(".gz"):
                file_to_remove = f"{file_name}.gz"

            success, message = await doc_proc.remove_file_from_stage(
                client, file_to_remove
            )

            if success:
                yield (
                    f"Successfully removed document '{file_name}' and any associated parsed content.",
                    {"success": True, "file_name": file_name, "message": message},
                )
                return
            else:
                yield (
                    f"Failed to remove document '{file_name}': {message}",
                    {"success": False, "error": message},
                )
                return

        except Exception as e:
            error_msg = f"Error removing document '{file_name}': {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "search_document":
        query = tool_args.get("query", "")
        file_name = tool_args.get("file_name")
        top_k = tool_args.get("top_k", 5)

        if not query:
            yield "Error: query is required", {"error": "query required"}
            return

        try:
            doc_proc = DocumentProcessor.instance()
            conv_id = tool_args.get("_conversation_id", "default")

            # Check if user is searching for tables specifically
            is_table_search = any(
                word in query.lower()
                for word in ["table", "tables", "tabular", "grid", "matrix"]
            )

            # STEP 0: For table searches, query DOCUMENT_PARSE_RESULTS for pages with "|"
            # then extract/parse actual table structure from those results
            if is_table_search:
                yield to_sse(
                    reasoning_step(
                        "Querying parsed documents for pages containing tables...",
                        event_type="INFO",
                    )
                )

                try:
                    snowflake_user = await asyncio.to_thread(client.get_current_user)
                    sanitized_user = "".join(
                        c if c.isalnum() else "_" for c in snowflake_user
                    )
                    user_schema = f"USER_{sanitized_user}".upper()

                    file_filter = ""
                    if file_name:
                        file_name_escaped = file_name.replace("'", "''")
                        file_filter = f"AND FILE_NAME = '{file_name_escaped}'"

                    # Query for pages with pipe characters (markdown table syntax)
                    table_sql = f"""
                    SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
                    FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS
                    WHERE PAGE_CONTENT LIKE '%|%|%'
                    {file_filter}
                    ORDER BY PAGE_NUMBER
                    """

                    result = await asyncio.to_thread(client.execute_query, table_sql)
                    result_json = json.loads(result)
                    row_data = result_json.get("rowData", [])

                    if row_data:
                        yield to_sse(
                            reasoning_step(
                                f"Found {len(row_data)} pages containing tables",
                                event_type="INFO",
                            )
                        )

                        # Just return the raw page content - it already has the tables!
                        output = "**Tables Found in Document**\n\n"

                        pages_with_tables = []
                        for row in row_data:
                            page_content = row.get(
                                "PAGE_CONTENT", row.get("page_content", "")
                            )
                            page_num = row.get(
                                "PAGE_NUMBER", row.get("page_number", "?")
                            )
                            fname = row.get(
                                "FILE_NAME", row.get("file_name", "Unknown")
                            )

                            output += f"---\n## Page {page_num} ({fname})\n\n"
                            output += page_content
                            output += "\n\n"

                            pages_with_tables.append(
                                {
                                    "page_number": page_num,
                                    "file_name": fname,
                                }
                            )

                        yield output, {
                            "results": pages_with_tables,
                            "query": query,
                            "search_type": "table_extraction",
                            "result_count": len(row_data),
                        }
                        return

                except Exception as table_err:
                    logger.warning("Table extraction failed: %s", table_err)

            # STEP 1: Search PDF metadata (outline/TOC, page summaries)
            # This is instant - no DB query needed
            metadata_results = doc_proc.search_pdf_metadata(conv_id, query, file_name)

            if metadata_results:
                yield to_sse(
                    reasoning_step(
                        f"Found {len(metadata_results)} matching sections in document structure",
                        event_type="INFO",
                    )
                )
                # Return metadata results - these point to specific pages
                output = f"**Document Structure Search for:** '{query}'\n\n"
                output += f"Found {len(metadata_results)} matching sections in document outline/TOC:\n\n"

                for i, match in enumerate(metadata_results, 1):
                    page = match.get("page_number", "?")
                    chunk_text = match.get("chunk_text", "")
                    match_type = match.get("match_type", "")

                    output += f"---\n**Result {i}** (Page {page}) [{match_type}]\n"
                    output += f"{chunk_text}\n\n"

                output += "\n💡 **Tip:** Use `read_document` with these page numbers to get full content."

                yield output, {
                    "results": metadata_results,
                    "query": query,
                    "search_type": "pdf_metadata",
                    "result_count": len(metadata_results),
                }
                return

            # STEP 2: Try semantic search (vector similarity)
            results, final_threshold = await doc_proc.semantic_search_documents(
                client,
                query=query,
                file_name=file_name,
                top_k=top_k,
                similarity_threshold=0.7,
            )

            # STEP 3: If semantic search returns no results, try keyword/text search
            if not results:
                yield to_sse(
                    reasoning_step(
                        f"Semantic search found no results. Trying keyword search for '{query}'...",
                        event_type="INFO",
                    )
                )

                # Fallback to keyword search in DOCUMENT_PARSE_RESULTS
                try:
                    snowflake_user = await asyncio.to_thread(client.get_current_user)
                    sanitized_user = "".join(
                        c if c.isalnum() else "_" for c in snowflake_user
                    )
                    user_schema = f"USER_{sanitized_user}".upper()

                    query_escaped = query.replace("'", "''")
                    file_filter = ""
                    if file_name:
                        file_name_escaped = file_name.replace("'", "''")
                        file_filter = f"AND FILE_NAME = '{file_name_escaped}'"

                    # Keyword search using ILIKE
                    keyword_sql = f"""
                    SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
                    FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS
                    WHERE LOWER(PAGE_CONTENT) LIKE LOWER('%{query_escaped}%')
                    {file_filter}
                    ORDER BY PAGE_NUMBER
                    LIMIT {top_k * 2}
                    """

                    result = await asyncio.to_thread(client.execute_query, keyword_sql)
                    result_json = json.loads(result)
                    row_data = result_json.get("rowData", [])

                    if row_data:
                        # Convert to results format
                        results = []
                        for row in row_data:
                            page_content = row.get(
                                "PAGE_CONTENT", row.get("page_content", "")
                            )
                            # Find snippet around the keyword
                            lower_content = page_content.lower()
                            lower_query = query.lower()
                            pos = lower_content.find(lower_query)
                            if pos >= 0:
                                start = max(0, pos - 200)
                                end = min(len(page_content), pos + len(query) + 300)
                                snippet = page_content[start:end]
                                if start > 0:
                                    snippet = "..." + snippet
                                if end < len(page_content):
                                    snippet = snippet + "..."
                            else:
                                snippet = (
                                    page_content[:500] + "..."
                                    if len(page_content) > 500
                                    else page_content
                                )

                            results.append(
                                {
                                    "file_name": row.get(
                                        "FILE_NAME", row.get("file_name", "Unknown")
                                    ),
                                    "page_number": row.get(
                                        "PAGE_NUMBER", row.get("page_number", "?")
                                    ),
                                    "chunk_text": snippet,
                                    "similarity_score": 1.0,  # Exact keyword match
                                    "match_type": "keyword",
                                }
                            )

                        yield to_sse(
                            reasoning_step(
                                f"Keyword search found {len(results)} pages containing '{query}'",
                                event_type="INFO",
                            )
                        )
                        final_threshold = 1.0  # Keyword match
                except Exception as keyword_err:
                    logger.warning("Keyword search fallback failed: %s", keyword_err)

            if not results:
                yield to_sse(
                    reasoning_step(
                        f"No matching document chunks found for query: '{query[:50]}...' (tried semantic and keyword search)",
                        event_type="WARNING",
                    )
                )
                yield (
                    f"No document content found matching '{query}'. Try rephrasing your search or use the read_document tool to view full document content.",
                    {"results": [], "query": query, "threshold_used": final_threshold},
                )
                return

            # Notify if threshold was lowered
            if final_threshold < 0.7:
                yield to_sse(
                    reasoning_step(
                        f"Lowered similarity threshold to {final_threshold:.2f} to find results",
                        event_type="INFO",
                    )
                )

            # Format results for output
            output = f"**Search Results for:** '{query}'\n\n"
            output += f"Found {len(results)} matching chunks (similarity threshold: {final_threshold:.2f})\n\n"

            for i, match in enumerate(results, 1):
                score = match.get("similarity_score", 0)
                fname = match.get("file_name", "Unknown")
                page = match.get("page_number", "?")
                content_type = match.get("content_type", "text")
                chunk_text = match.get("chunk_text", "")
                image_stage_path = match.get("image_stage_path")

                # Truncate long chunks for display
                if len(chunk_text) > 500:
                    chunk_text = chunk_text[:500] + "..."

                output += f"---\n**Result {i}** (Score: {score:.3f})\n"
                output += f"📄 File: {fname} | Page: {page}"

                if content_type == "image":
                    output += f" | 🖼️ **IMAGE**\n"
                    output += f"**Image Location:** `{image_stage_path}`\n"
                    output += f"**Page Context:** {chunk_text}\n\n"
                else:
                    output += f"\n\n{chunk_text}\n\n"

            yield output, {
                "results": results,
                "query": query,
                "threshold_used": final_threshold,
                "result_count": len(results),
            }
            return

        except Exception as e:
            error_msg = f"Error searching documents: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "get_document_images":
        file_name = tool_args.get("file_name", "")
        page_numbers = tool_args.get("page_numbers", [])

        if not file_name:
            yield "Error: file_name is required", {"error": "file_name required"}
            return

        try:
            # Get user schema
            snowflake_user = await asyncio.to_thread(client.get_current_user)
            sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
            user_schema = f"USER_{sanitized_user}".upper()

            file_name_escaped = file_name.replace("'", "''")

            # Build page filter if provided
            page_filter = ""
            page_filter_m = ""  # For metadata table (m alias)
            page_filter_e = ""  # For embeddings table (no alias)
            if page_numbers:
                page_list = ",".join(str(p) for p in page_numbers)
                page_filter_m = f"AND m.PAGE_NUMBER IN ({page_list})"
                page_filter_e = f"AND PAGE_NUMBER IN ({page_list})"

            # First try DOCUMENT_IMAGES_METADATA (uploaded images awaiting/after embedding)
            # Then fall back to DOCUMENT_EMBEDDINGS (embedded images)
            query_sql = f"""
            SELECT 
                COALESCE(e.EMBEDDING_ID, m.ID) as EMBEDDING_ID,
                COALESCE(m.FILE_NAME, e.FILE_NAME) as FILE_NAME,
                COALESCE(m.PAGE_NUMBER, e.PAGE_NUMBER) as PAGE_NUMBER,
                COALESCE(m.IMAGE_INDEX, e.CHUNK_INDEX) as IMAGE_INDEX,
                COALESCE(m.IMAGE_STAGE_PATH, e.IMAGE_STAGE_PATH) as IMAGE_STAGE_PATH,
                e.CHUNK_TEXT as PAGE_CONTEXT,
                COALESCE(m.IMAGE_HASH, e.IMAGE_HASH) as IMAGE_HASH,
                m.IMAGE_FORMAT,
                m.WIDTH,
                m.HEIGHT,
                CASE WHEN e.EMBEDDING_ID IS NOT NULL THEN TRUE ELSE FALSE END as HAS_EMBEDDING
            FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_IMAGES_METADATA m
            LEFT JOIN OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS e
                ON m.FILE_NAME = e.FILE_NAME 
                AND m.PAGE_NUMBER = e.PAGE_NUMBER 
                AND m.IMAGE_INDEX = e.CHUNK_INDEX
                AND e.CONTENT_TYPE = 'image'
            WHERE m.FILE_NAME = '{file_name_escaped}'
              {page_filter_m}
            ORDER BY m.PAGE_NUMBER, m.IMAGE_INDEX
            """

            result = await asyncio.to_thread(client.execute_statement, query_sql)

            # If metadata table doesn't exist or is empty, try embeddings table directly
            rows = []
            if result:
                rows = json.loads(result) if isinstance(result, str) else result
                if isinstance(rows, dict):
                    rows = rows.get("data", rows.get("DATA", []))

            if not rows:
                # Fallback: query embeddings table directly
                fallback_sql = f"""
                SELECT 
                    EMBEDDING_ID,
                    FILE_NAME,
                    PAGE_NUMBER,
                    CHUNK_INDEX as IMAGE_INDEX,
                    IMAGE_STAGE_PATH,
                    CHUNK_TEXT as PAGE_CONTEXT,
                    IMAGE_HASH,
                    NULL as IMAGE_FORMAT,
                    NULL as WIDTH,
                    NULL as HEIGHT,
                    TRUE as HAS_EMBEDDING
                FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS
                WHERE FILE_NAME = '{file_name_escaped}'
                  AND CONTENT_TYPE = 'image'
                  {page_filter_e}
                ORDER BY PAGE_NUMBER, CHUNK_INDEX
                """
                result = await asyncio.to_thread(client.execute_statement, fallback_sql)
                if result:
                    rows = json.loads(result) if isinstance(result, str) else result
                    if isinstance(rows, dict):
                        rows = rows.get("data", rows.get("DATA", []))

            if not rows:
                page_info = f" on pages {page_numbers}" if page_numbers else ""
                yield f"No images found for document '{file_name}'{page_info}", {
                    "images": []
                }
                return

            # Format output
            output = f"**Images from:** {file_name}\n\n"
            output += f"Found {len(rows)} images\n\n"

            images = []
            for row in rows:
                if isinstance(row, (list, tuple)):
                    img_data = {
                        "embedding_id": row[0],
                        "file_name": row[1],
                        "page_number": row[2],
                        "image_index": row[3],
                        "image_stage_path": row[4],
                        "page_context": row[5],
                        "image_hash": row[6],
                        "image_format": row[7],
                        "width": row[8],
                        "height": row[9],
                        "has_embedding": row[10],
                    }
                else:
                    img_data = {
                        "embedding_id": row.get("EMBEDDING_ID")
                        or row.get("embedding_id"),
                        "file_name": row.get("FILE_NAME") or row.get("file_name"),
                        "page_number": row.get("PAGE_NUMBER") or row.get("page_number"),
                        "image_index": row.get("IMAGE_INDEX")
                        or row.get("image_index")
                        or row.get("CHUNK_INDEX")
                        or row.get("chunk_index"),
                        "image_stage_path": row.get("IMAGE_STAGE_PATH")
                        or row.get("image_stage_path"),
                        "page_context": row.get("PAGE_CONTEXT")
                        or row.get("page_context")
                        or row.get("CHUNK_TEXT")
                        or row.get("chunk_text"),
                        "image_hash": row.get("IMAGE_HASH") or row.get("image_hash"),
                        "image_format": row.get("IMAGE_FORMAT")
                        or row.get("image_format"),
                        "width": row.get("WIDTH") or row.get("width"),
                        "height": row.get("HEIGHT") or row.get("height"),
                        "has_embedding": row.get("HAS_EMBEDDING")
                        or row.get("has_embedding"),
                    }

                images.append(img_data)

                output += f"---\n"
                output += f"**Image {img_data['image_index']}** | Page {img_data['page_number']}"
                if img_data.get("has_embedding"):
                    output += " ✓ embedded"
                output += "\n"
                output += f"📍 Stage Path: `{img_data['image_stage_path']}`\n"
                if img_data.get("width") and img_data.get("height"):
                    output += f"📐 Size: {img_data['width']}x{img_data['height']}\n"
                if img_data["page_context"]:
                    context = (
                        img_data["page_context"][:200] + "..."
                        if len(str(img_data["page_context"])) > 200
                        else img_data["page_context"]
                    )
                    output += f"📝 Page Context: {context}\n"
                output += "\n"

            yield output, {
                "images": images,
                "file_name": file_name,
                "image_count": len(images),
            }
            return

        except Exception as e:
            error_msg = f"Error getting document images: {str(e)}"
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "ocr_image":
        image_stage_path = tool_args.get("image_stage_path")
        file_name = tool_args.get("file_name")
        page_number = tool_args.get("page_number")
        extract_tables = tool_args.get("extract_tables", True)

        try:
            # Get user schema
            snowflake_user = await asyncio.to_thread(client.get_current_user)
            sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
            user_schema = f"USER_{sanitized_user}".upper()

            # If file_name and page_number provided, look up the image path
            # Try DOCUMENT_IMAGES_METADATA first (rendered page images), then DOCUMENT_EMBEDDINGS
            if not image_stage_path and file_name and page_number:
                file_name_escaped = file_name.replace("'", "''")
                logger.info(
                    f"[ocr_image] Looking up image for file={file_name}, page={page_number}"
                )

                # Try metadata table first (has rendered page images)
                try:
                    metadata_sql = f"""
                    SELECT IMAGE_STAGE_PATH
                    FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_IMAGES_METADATA
                    WHERE FILE_NAME = '{file_name_escaped}'
                      AND PAGE_NUMBER = {page_number}
                    ORDER BY IMAGE_INDEX
                    LIMIT 1
                    """
                    result = await asyncio.to_thread(client.execute_query, metadata_sql)
                    if result:
                        result_json = (
                            json.loads(result) if isinstance(result, str) else result
                        )
                        rows = result_json.get("rowData", [])
                        if rows:
                            row = rows[0]
                            image_stage_path = row.get("IMAGE_STAGE_PATH") or row.get(
                                "image_stage_path"
                            )
                            logger.info(
                                f"[ocr_image] Found in DOCUMENT_IMAGES_METADATA: {image_stage_path}"
                            )
                        else:
                            logger.info(
                                f"[ocr_image] DOCUMENT_IMAGES_METADATA returned no rows"
                            )
                except Exception as meta_err:
                    logger.warning(
                        f"[ocr_image] DOCUMENT_IMAGES_METADATA lookup failed: {meta_err}"
                    )

                # Fall back to embeddings table if metadata didn't work
                if not image_stage_path:
                    embeddings_sql = f"""
                    SELECT IMAGE_STAGE_PATH
                    FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS
                    WHERE FILE_NAME = '{file_name_escaped}'
                      AND CONTENT_TYPE = 'image'
                      AND PAGE_NUMBER = {page_number}
                    ORDER BY CHUNK_INDEX
                    LIMIT 1
                    """
                    result = await asyncio.to_thread(
                        client.execute_query, embeddings_sql
                    )
                    if result:
                        result_json = (
                            json.loads(result) if isinstance(result, str) else result
                        )
                        rows = result_json.get("rowData", [])
                        if rows:
                            row = rows[0]
                            image_stage_path = row.get("IMAGE_STAGE_PATH") or row.get(
                                "image_stage_path"
                            )

                # Final fallback: try to list files in stage directly to find the image
                if not image_stage_path:
                    try:
                        # List files in the document's folder in DOCUMENT_IMAGES stage
                        list_sql = f"""
                        LIST @OPENBB_AGENTS.{user_schema}.DOCUMENT_IMAGES/{file_name_escaped}/
                        """
                        result = await asyncio.to_thread(
                            client.execute_statement, list_sql
                        )
                        if result:
                            rows = (
                                json.loads(result)
                                if isinstance(result, str)
                                else result
                            )
                            if isinstance(rows, dict):
                                rows = rows.get(
                                    "data", rows.get("DATA", rows.get("rowData", []))
                                )

                            # Find image for this page number
                            for row in rows:
                                # LIST returns "name" column with stage path like:
                                # document_images/Bombardier.pdf/page_5_image_8.jpeg
                                if isinstance(row, dict):
                                    file_path = row.get("name", row.get("NAME", ""))
                                elif isinstance(row, (list, tuple)):
                                    file_path = row[0] if row else ""
                                else:
                                    continue

                                # Match pattern: page_X_image_Y.ext
                                if f"page_{page_number}_" in file_path:
                                    # Extract just the filename part
                                    img_filename = file_path.split("/")[-1]
                                    image_stage_path = f"@OPENBB_AGENTS.{user_schema}.DOCUMENT_IMAGES/{file_name}/{img_filename}"
                                    logger.info(
                                        f"Found image via stage LIST: {image_stage_path}"
                                    )
                                    break
                    except Exception as list_err:
                        logger.warning(f"Stage LIST fallback failed: {list_err}")

            if not image_stage_path:
                yield "Error: No image found. Provide image_stage_path or valid file_name + page_number", {
                    "error": "no image"
                }
                return

            # Parse the stage path to extract stage and filename for TO_FILE
            # Format: @DATABASE.SCHEMA.STAGE/path/to/file.jpg
            if image_stage_path.startswith("@"):
                path_parts = image_stage_path[1:].split("/", 1)
                stage_name = f"@{path_parts[0]}"
                file_path = path_parts[1] if len(path_parts) > 1 else ""
            else:
                stage_name = image_stage_path
                file_path = ""

            file_path_escaped = file_path.replace("'", "''")
            return_as_chart = tool_args.get("return_as_chart", False)

            # Use SNOWFLAKE.CORTEX.COMPLETE with vision model for accurate chart extraction
            # Vision models can see spatial relationships between X-axis labels and values
            vision_prompt = """Analyze this image carefully. If it contains a chart, graph, or data visualization:

1. Identify the chart type (bar, line, pie, scatter, etc.)
2. Check if the image contains MULTIPLE separate chart sections/panels with different metrics
3. For each section, read the X-axis labels from LEFT to RIGHT
4. For each X-axis label, identify the corresponding value DIRECTLY ABOVE/AT that position
5. Extract the Y-axis units/label if visible for each section (e.g., "$M", "USD millions", "%")
6. Extract the chart title if visible

If the image contains MULTIPLE chart sections with different metrics (e.g., "Revenue", "Profit", "EPS" as separate panels), return:
{
  "chart_type": "bar|line|pie|scatter|other",
  "title": "overall chart title or null",
  "data": [
    {"section": "Metric Name (units)", "unit": "$M", "data": [{"label": "Q1 2024", "value": 100}, {"label": "Q1 2025", "value": 120}]},
    {"section": "Another Metric (units)", "unit": "$", "data": [{"label": "Q1 2024", "value": 0.50}, {"label": "Q1 2025", "value": 0.60}]}
  ]
}

IMPORTANT for section names:
- Include the metric name AND units in the section name, e.g., "Adjusted Net Income ($M)", "Adjusted EPS ($)", "Free Cash Flow ($M)"
- If units are shown near the values (like "$68M"), include them in the section name
- The "unit" field should contain just the unit symbol like "$M", "$", "%", etc.

If the image contains a SINGLE chart section, return:
{
  "chart_type": "bar|line|pie|scatter|other",
  "title": "chart title or null",
  "x_label": "x-axis label or null",
  "y_label": "y-axis label with units or null", 
  "unit": "unit symbol like $M, $, %, etc.",
  "data": [{"label": "x-axis value", "value": numeric_value}, ...]
}

If the image is NOT a chart, return:
{
  "chart_type": null,
  "description": "natural language description of what you see",
  "text_content": "any text visible in the image"
}

CRITICAL: 
- Read X-axis labels LEFT to RIGHT
- Match each label to its corresponding bar/line value by vertical alignment
- ALWAYS include units in section names (e.g., "Revenue ($M)" not just "Revenue")
- If multiple chart sections exist, group them by section name with units
- Do NOT guess - if you cannot read a value clearly, use null"""

            vision_prompt_escaped = vision_prompt.replace("'", "''")

            vision_sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'claude-3-5-sonnet',
                '{vision_prompt_escaped}',
                TO_FILE('{stage_name}', '{file_path_escaped}')
            ) AS VISION_RESULT
            """

            yield to_sse(
                reasoning_step(
                    f"Analyzing image with vision model: {image_stage_path}",
                    event_type="INFO",
                )
            )

            result = None
            vision_error = None
            try:
                logger.info(
                    f"[ocr_image] Calling vision model with SQL: {vision_sql[:200]}..."
                )
                result = await asyncio.to_thread(client.execute_statement, vision_sql)
                logger.info(
                    f"[ocr_image] Vision model returned: {str(result)[:500] if result else 'None'}"
                )
            except Exception as vision_exc:
                vision_error = str(vision_exc)
                logger.error(f"[ocr_image] Vision model failed: {vision_error}")

            # Parse vision model result
            extracted_data = None
            extracted_text = ""
            chart_type = None

            if result:
                rows = json.loads(result) if isinstance(result, str) else result
                if isinstance(rows, dict):
                    rows = rows.get("data", rows.get("DATA", []))

                if rows:
                    row = rows[0]
                    if isinstance(row, (list, tuple)):
                        vision_result = row[0]
                    else:
                        vision_result = row.get("VISION_RESULT") or row.get(
                            "vision_result"
                        )

                    # Parse the vision result (should be JSON)
                    if isinstance(vision_result, str):
                        # Try to extract JSON from the response (model may add explanation text)
                        try:
                            # Try direct parse first
                            extracted_data = json.loads(vision_result)
                        except json.JSONDecodeError:
                            # Try to find JSON in the response
                            import re

                            json_match = re.search(
                                r'\{[^{}]*"(?:chart_type|data|description)"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',
                                vision_result,
                                re.DOTALL,
                            )
                            if json_match:
                                try:
                                    extracted_data = json.loads(json_match.group())
                                except json.JSONDecodeError:
                                    extracted_text = vision_result
                            else:
                                extracted_text = vision_result
                    elif isinstance(vision_result, dict):
                        extracted_data = vision_result

            # If vision failed completely, fall back to OCR
            if not extracted_data and not extracted_text:
                logger.info(
                    "[ocr_image] Vision model returned no usable data, falling back to OCR"
                )
                ocr_sql = f"""
                SELECT AI_PARSE_DOCUMENT(
                    TO_FILE('{stage_name}', '{file_path_escaped}'),
                    {{'mode': 'OCR'}}
                ) AS OCR_RESULT
                """
                try:
                    logger.info(f"[ocr_image] Calling OCR with SQL: {ocr_sql[:200]}...")
                    ocr_result = await asyncio.to_thread(
                        client.execute_statement, ocr_sql
                    )
                    logger.info(
                        f"[ocr_image] OCR returned: {str(ocr_result)[:500] if ocr_result else 'None'}"
                    )
                    if ocr_result:
                        ocr_rows = (
                            json.loads(ocr_result)
                            if isinstance(ocr_result, str)
                            else ocr_result
                        )
                        if isinstance(ocr_rows, dict):
                            ocr_rows = ocr_rows.get("data", ocr_rows.get("DATA", []))
                        if ocr_rows:
                            ocr_row = ocr_rows[0]
                            if isinstance(ocr_row, (list, tuple)):
                                ocr_text = ocr_row[0]
                            else:
                                ocr_text = ocr_row.get("OCR_RESULT") or ocr_row.get(
                                    "ocr_result"
                                )
                            if isinstance(ocr_text, str):
                                try:
                                    ocr_parsed = json.loads(ocr_text)
                                    extracted_text = ocr_parsed.get(
                                        "content",
                                        ocr_parsed.get("text", str(ocr_parsed)),
                                    )
                                except json.JSONDecodeError:
                                    extracted_text = ocr_text
                            elif isinstance(ocr_text, dict):
                                extracted_text = ocr_text.get(
                                    "content",
                                    ocr_text.get("text", json.dumps(ocr_text)),
                                )
                except Exception as ocr_exc:
                    logger.error(f"[ocr_image] OCR fallback also failed: {ocr_exc}")
                    extracted_text = f"Unable to extract content from image. Vision error: {vision_error or 'no result'}. OCR error: {ocr_exc}"

            # ALWAYS return something - never exit without output
            if not extracted_data and not extracted_text:
                extracted_text = f"Image found at {image_stage_path} but could not extract content. Vision error: {vision_error or 'unknown'}. Please try read_document_page to get the text content of this page."
                logger.warning(
                    f"[ocr_image] No extraction succeeded, returning fallback message"
                )

            # Process extracted data for output
            if extracted_data and isinstance(extracted_data, dict):
                chart_type = extracted_data.get("chart_type")
                raw_data_points = extracted_data.get("data", [])

                # Handle nested data structure from vision model
                # Format 1: [{"label": "x", "value": y}, ...]
                # Format 2: [{"section": "name", "data": [{"label": "x", "value": y}, ...]}, ...]
                data_points = []
                sections = []
                for dp in raw_data_points:
                    if isinstance(dp, dict):
                        if "section" in dp and "data" in dp:
                            # Nested format - flatten with section prefix
                            section_name = dp.get("section", "")
                            sections.append(section_name)
                            for inner_dp in dp.get("data", []):
                                if (
                                    isinstance(inner_dp, dict)
                                    and inner_dp.get("label") is not None
                                ):
                                    # Add section context to label
                                    data_points.append(
                                        {
                                            "label": f"{inner_dp.get('label')} ({section_name})",
                                            "value": inner_dp.get("value"),
                                            "section": section_name,
                                            "original_label": inner_dp.get("label"),
                                        }
                                    )
                        elif dp.get("label") is not None:
                            # Simple format
                            data_points.append(dp)

                logger.info(
                    f"[ocr_image] Parsed {len(data_points)} data points from {len(sections) or 1} section(s)"
                )
                print(
                    f"[ocr_image] Parsed {len(data_points)} data points from {len(sections)} sections: {sections}"
                )
                print(
                    f"[ocr_image] Data points sample: {data_points[:3]}{'...' if len(data_points) > 3 else ''}"
                )

                if chart_type and data_points and return_as_chart:
                    # Return as OpenBB chart artifact
                    try:
                        from openbb_ai.helpers import chart, table

                        # Determine chart type based on content
                        # Time series with enough points -> line, otherwise bar
                        openbb_chart_type = "bar"  # default
                        if chart_type == "line":
                            openbb_chart_type = "line"
                        elif chart_type == "pie":
                            openbb_chart_type = "pie"
                        elif chart_type == "scatter":
                            openbb_chart_type = "scatter"
                        # For bar charts or unknown, keep as bar

                        chart_title = (
                            extracted_data.get("title") or "Extracted Chart Data"
                        )
                        y_label = extracted_data.get("y_label") or "Value"

                        # Check if we have multiple sections (each needs its own chart for proper scaling)
                        if sections and len(sections) > 1:
                            # Group data by section and create separate charts
                            section_data = {}
                            for dp in data_points:
                                if isinstance(dp, dict) and dp.get("value") is not None:
                                    section = dp.get("section", "")
                                    if section not in section_data:
                                        section_data[section] = []
                                    section_data[section].append(
                                        {
                                            "label": str(
                                                dp.get(
                                                    "original_label",
                                                    dp.get("label", ""),
                                                )
                                            ),
                                            "value": (
                                                float(dp["value"])
                                                if dp["value"] is not None
                                                else 0
                                            ),
                                        }
                                    )

                            # Create a chart for each section
                            all_charts_summary = (
                                f"**Charts extracted from:** `{image_stage_path}`\n\n"
                            )
                            charts_created = 0

                            print(
                                f"[ocr_image] Creating charts for {len(section_data)} sections: {list(section_data.keys())}"
                            )

                            for section_name, section_points in section_data.items():
                                if len(section_points) >= 1:
                                    # Ensure section name is not empty
                                    display_name = (
                                        section_name.strip()
                                        if section_name
                                        else f"Series {charts_created + 1}"
                                    )

                                    # Use display_name as the value key for proper axis labeling
                                    # Transform data to use display_name as the y-axis key
                                    chart_points = []
                                    for pt in section_points:
                                        chart_points.append(
                                            {
                                                "Quarter": str(pt.get("label", "")),
                                                display_name: pt.get("value", 0),
                                            }
                                        )

                                    print(
                                        f"[ocr_image] Creating chart '{display_name}' with {len(chart_points)} points"
                                    )

                                    if openbb_chart_type in ("pie", "donut"):
                                        section_chart = chart(
                                            type=openbb_chart_type,
                                            data=chart_points,
                                            angle_key=display_name,
                                            callout_label_key="Quarter",
                                            name=display_name,
                                            description=f"Extracted from: {image_stage_path}",
                                        )
                                    else:
                                        section_chart = chart(
                                            type=openbb_chart_type,
                                            data=chart_points,
                                            x_key="Quarter",
                                            y_keys=[display_name],
                                            name=display_name,
                                            description=f"Extracted from: {image_stage_path}",
                                        )
                                    yield to_sse(section_chart)
                                    charts_created += 1

                                    all_charts_summary += f"**{display_name}:**\n"
                                    for dp in section_points:
                                        all_charts_summary += (
                                            f"- {dp['label']}: {dp['value']}\n"
                                        )
                                    all_charts_summary += "\n"

                            if charts_created > 0:
                                yield all_charts_summary, {
                                    "image_stage_path": image_stage_path,
                                    "chart_type": chart_type,
                                    "sections": list(section_data.keys()),
                                    "data": section_data,
                                    "title": chart_title,
                                    "extracted_data": extracted_data,
                                }
                                return

                        # Single section or no sections - create one combined chart
                        # Build chart data with proper axis labels
                        chart_data = []
                        x_axis_label = extracted_data.get("x_label") or "Category"
                        y_axis_label = y_label if y_label != "Value" else chart_title

                        for dp in data_points:
                            if (
                                isinstance(dp, dict)
                                and dp.get("label") is not None
                                and dp.get("value") is not None
                            ):
                                chart_data.append(
                                    {
                                        x_axis_label: str(dp["label"]),
                                        y_axis_label: (
                                            float(dp["value"])
                                            if dp["value"] is not None
                                            else 0
                                        ),
                                    }
                                )

                        if len(chart_data) >= 2:
                            # Return chart artifact
                            if openbb_chart_type in ("pie", "donut"):
                                chart_artifact = chart(
                                    type=openbb_chart_type,
                                    data=chart_data,
                                    angle_key=y_axis_label,
                                    callout_label_key=x_axis_label,
                                    name=chart_title,
                                    description=f"Extracted from: {image_stage_path}",
                                )
                            else:
                                chart_artifact = chart(
                                    type=openbb_chart_type,
                                    data=chart_data,
                                    x_key=x_axis_label,
                                    y_keys=[y_axis_label],
                                    name=chart_title,
                                    description=f"Extracted from: {image_stage_path}",
                                )
                            # Yield the chart artifact as SSE event for the UI
                            yield to_sse(chart_artifact)

                            # Build descriptive text summary for the LLM
                            summary_text = (
                                f"**Chart extracted from:** `{image_stage_path}`\n"
                            )
                            summary_text += f"**Chart type:** {openbb_chart_type}\n"
                            summary_text += f"**Title:** {chart_title}\n"
                            summary_text += f"**Data points:** {len(chart_data)} values extracted\n\n"
                            for dp in chart_data:
                                summary_text += (
                                    f"- {dp[x_axis_label]}: {dp[y_axis_label]}\n"
                                )

                            yield summary_text, {
                                "image_stage_path": image_stage_path,
                                "chart_type": chart_type,
                                "data": chart_data,
                                "title": chart_title,
                                "extracted_data": extracted_data,
                            }
                            return
                        elif len(chart_data) == 1:
                            # Single data point - return as text
                            dp = chart_data[0]
                            output = f"**Extracted from chart:** {dp['label']}: {dp['value']} {y_label}\n"
                            yield output, {
                                "image_stage_path": image_stage_path,
                                "extracted_data": extracted_data,
                            }
                            return
                    except ImportError:
                        logger.warning(
                            "openbb_ai.helpers not available for chart output"
                        )
                    except Exception as chart_exc:
                        logger.warning(f"Failed to create chart artifact: {chart_exc}")

                # Format structured data as text output
                output = f"**Chart Analysis for:** `{image_stage_path}`\n\n"
                if extracted_data.get("title"):
                    output += f"**Title:** {extracted_data['title']}\n"
                if chart_type:
                    output += f"**Chart Type:** {chart_type}\n"
                if extracted_data.get("y_label"):
                    output += f"**Y-Axis:** {extracted_data['y_label']}\n"
                if extracted_data.get("x_label"):
                    output += f"**X-Axis:** {extracted_data['x_label']}\n"

                if data_points:
                    output += "\n**Data Points:**\n"
                    for dp in data_points:
                        if isinstance(dp, dict):
                            label = dp.get("label", "Unknown")
                            value = dp.get("value", "N/A")
                            output += f"- {label}: {value}\n"
                        else:
                            output += f"- {dp}\n"
                elif extracted_data.get("description"):
                    output += f"\n**Description:** {extracted_data['description']}\n"
                if extracted_data.get("text_content"):
                    output += f"\n**Text Content:**\n{extracted_data['text_content']}\n"

                yield output, {
                    "image_stage_path": image_stage_path,
                    "chart_type": chart_type,
                    "extracted_data": extracted_data,
                }
                return

            # Fallback: return raw text
            output = f"**Image Analysis for:** `{image_stage_path}`\n\n"
            output += f"---\n{extracted_text}\n---\n"

            yield output, {
                "image_stage_path": image_stage_path,
                "extracted_text": extracted_text,
            }
            return

        except Exception as e:
            error_msg = f"Error performing OCR: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "ai_filter":
        predicate = tool_args.get("predicate", "")
        text = tool_args.get("text")
        query = tool_args.get("query")
        column_name = tool_args.get("column_name")
        image_stage_path = tool_args.get("image_stage_path")

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
                result = await asyncio.to_thread(client.execute_query, filter_sql)
                result_json = json.loads(result)

                row_data = result_json.get("rowData", [])
                if row_data:
                    ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                    output = f"**AI Filter Result:**\n\n"
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
                result = await asyncio.to_thread(client.execute_query, filter_sql)
                result_json = json.loads(result)

                row_data = result_json.get("rowData", [])
                if row_data:
                    ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                    output = f"**AI Filter Result (Image):**\n\n"
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

                result = await asyncio.to_thread(client.execute_query, filter_sql)
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
                output = f"**AI Filter Results:**\n\n"
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
                result = await asyncio.to_thread(client.execute_query, filter_sql)
                result_json = json.loads(result)

                row_data = result_json.get("rowData", [])
                if row_data:
                    ai_result = row_data[0].get("RESULT", row_data[0].get("result"))
                    output = f"**AI Filter Result:**\n\n"
                    output += f"**Question/Statement:** {predicate}\n\n"
                    output += f"**Result:** {'✅ TRUE' if ai_result else '❌ FALSE'}"
                    yield output, {"result": ai_result, "predicate": predicate}
                else:
                    yield "No result returned from AI_FILTER", {"error": "No result"}
                return

        except Exception as e:
            error_msg = f"Error executing AI_FILTER: {str(e)}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("AI_FILTER error: %s", traceback.format_exc())
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "ai_agg":
        import re

        instruction = tool_args.get("instruction", "")
        text = tool_args.get("text")
        query = tool_args.get("query")
        text_column = tool_args.get("text_column")

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
            result = await asyncio.to_thread(client.execute_query, ai_agg_sql)
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
                    col
                    for col in row_data[0].keys()
                    if col.upper() != "AGGREGATED_RESULT"
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
            return

        except Exception as e:
            error_msg = f"Error executing AI_AGG: {str(e)}"
            logger.error("AI_AGG error: %s", traceback.format_exc())
            yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
            yield error_msg, {"error": str(e)}
            return

    elif tool_name == "ai_summarize_agg":
        import re

        text = tool_args.get("text")
        query = tool_args.get("query")
        text_column = tool_args.get("text_column")

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
            result = await asyncio.to_thread(client.execute_query, summarize_sql)
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
                group_cols = [
                    col for col in row_data[0].keys() if col.upper() != "SUMMARY"
                ]

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
            return

        except Exception as e:
            error_msg = f"Error executing AI_SUMMARIZE_AGG: {str(e)}"
            logger.error("AI_SUMMARIZE_AGG error: %s", traceback.format_exc())
            yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
            yield error_msg, {"error": str(e)}
            return


def get_tool_definitions(client: SnowflakeAI) -> list:
    """Get the list of available tool definitions."""
    raw_tools = [
        client.text2sql_tool(),
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
            "description": "Read specific pages from an uploaded document. IMPORTANT: Always specify page_numbers to avoid context overflow! Use search_document first to find relevant pages, then read those specific pages. Maximum recommended: 5-10 pages per call.",
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
                        "description": "RECOMMENDED: List of specific page numbers to retrieve (e.g., [1, 5, 12]). Use search_document first to find relevant pages. Omitting this parameter loads ALL pages which may cause context overflow on large documents!",
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

    # Add search_document tool for semantic search across documents
    search_document_tool = {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Search uploaded documents using semantic similarity. Use this to find specific information, topics, or answers within documents. Returns the most relevant text chunks with similarity scores. More efficient than reading entire documents when looking for specific content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - describe what you're looking for in natural language (e.g., 'revenue growth projections', 'risk factors mentioned', 'CEO statement on market outlook').",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional: limit search to a specific document filename. If not provided, searches across all uploaded documents.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top matching chunks to return (default: 5, max: 20).",
                    },
                },
                "required": ["query"],
            },
        },
    }
    raw_tools.append(search_document_tool)

    # Add get_document_images tool for retrieving images from documents
    get_document_images_tool = {
        "type": "function",
        "function": {
            "name": "get_document_images",
            "description": "Get images from a document by page number. Returns image URLs that can be displayed or analyzed. Use after search_document finds relevant image results, or to get all images from specific pages. Images include charts, graphs, diagrams, and photos extracted from the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The document filename to get images from.",
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of page numbers to get images from (e.g., [1, 5, 10]). If not provided, returns all images.",
                    },
                },
                "required": ["file_name"],
            },
        },
    }
    raw_tools.append(get_document_images_tool)

    # Add ocr_image tool for extracting text from images
    # Uses vision model (claude-3-5-sonnet) for accurate chart/graph extraction
    ocr_image_tool = {
        "type": "function",
        "function": {
            "name": "ocr_image",
            "description": "Extract text and data from an image using AI vision. BEST FOR: reading charts, graphs, and visualizations where spatial relationships matter (correctly matching X-axis labels to values). Also works for tables, diagrams, and general images. Uses vision model for accurate chart extraction, with OCR fallback. Use after get_document_images to analyze specific images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_stage_path": {
                        "type": "string",
                        "description": "The stage path to the image (e.g., '@OPENBB_AGENTS.USER_DLEE.DOCUMENT_IMAGES/doc.pdf/page_5_image_0.jpeg'). Get this from get_document_images results.",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Alternative: document filename. If provided with page_number, will analyze that page's image.",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Page number to analyze (use with file_name instead of image_stage_path).",
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "If true, attempt to extract tabular data as structured format. Default: true.",
                    },
                    "return_as_chart": {
                        "type": "boolean",
                        "description": "If true and image contains a chart, return as interactive chart artifact. Chart type is determined by the extracted content. Default: false.",
                    },
                },
                "required": [],
            },
        },
    }
    raw_tools.append(ocr_image_tool)

    # Add ai_filter tool for boolean classification of text/data using AI
    ai_filter_tool = {
        "type": "function",
        "function": {
            "name": "ai_filter",
            "description": "Use AI to classify text/data as TRUE or FALSE based on a natural language condition. Best for: (1) Filtering query results by semantic meaning - e.g., filter reviews where 'customer sounds satisfied', (2) Yes/no classification of text - e.g., 'Is this about financial risk?', (3) Image classification - e.g., 'Is this a product photo?'. NOT for: extracting data (use extract_answer), summarizing (use summarize), or searching documents (use search_document).",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "description": "Natural language condition to evaluate as TRUE/FALSE. Phrase as statement: 'The customer sounds satisfied', 'This discusses financial risk', 'The sentiment is positive'. Be specific for accuracy.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to classify. Use for single text classification without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query whose results will be filtered. AI_FILTER evaluates each row against the predicate.",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Column name to apply AI filter to when using 'query'. Required with query parameter.",
                    },
                    "image_stage_path": {
                        "type": "string",
                        "description": "Stage path to image for classification (e.g., '@DB.SCHEMA.STAGE/img.jpg'). Use instead of text for images.",
                    },
                },
                "required": ["predicate"],
            },
        },
    }
    raw_tools.append(ai_filter_tool)

    # Add ai_agg tool for AI-powered text aggregation
    ai_agg_tool = {
        "type": "function",
        "function": {
            "name": "ai_agg",
            "description": "Reduce large text columns using natural language instructions. Handles datasets LARGER than LLM context windows (unlike summarize tool). Examples: AI_AGG(reviews, 'Summarize customer feedback'), AI_AGG('Menu: ' || menu_item || '\\nReview: ' || review, 'Find most positive review to highlight on website'), SELECT product_id, AI_AGG(review, 'Identify common complaints') FROM reviews GROUP BY product_id. Use for: aggregating reviews/comments/transcripts across many rows, extracting patterns from large datasets, custom aggregation instructions like 'Describe common complaints', 'Identify all people mentioned with short biographies', 'Find patterns across customer feedback'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction describing how to aggregate the text. Use declarative statements like 'Summarize the reviews', 'Identify common themes', 'Find the most positive review'. Be specific about the intended use case.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Direct text string to aggregate. Use for simple aggregations without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query that returns text data to aggregate. Can include GROUP BY to aggregate by groups. Example: 'SELECT product_id, review FROM reviews'.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column name containing text to aggregate when using 'query' parameter. Required when query is provided.",
                    },
                },
                "required": ["instruction"],
            },
        },
    }
    raw_tools.append(ai_agg_tool)

    # Add ai_summarize_agg tool for general-purpose text summarization
    ai_summarize_agg_tool = {
        "type": "function",
        "function": {
            "name": "ai_summarize_agg",
            "description": "General-purpose summarization of large text columns. Handles datasets LARGER than LLM context windows. Examples: AI_SUMMARIZE_AGG(churn_reason), SELECT restaurant_id, AI_SUMMARIZE_AGG(review) FROM reviews GROUP BY restaurant_id, AI_SUMMARIZE_AGG('Item: ' || item || '\\nReview: ' || text). Automatically generates summaries without needing custom instructions. For specific aggregations with custom prompts like 'identify complaints' or 'find patterns', use ai_agg instead. For single document summarization, use summarize tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Direct text string to summarize. Use for simple summarizations without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query that returns text data to summarize. Can include GROUP BY to summarize by groups.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column name containing text to summarize when using 'query' parameter. Required when query is provided.",
                    },
                },
                "required": [],
            },
        },
    }
    raw_tools.append(ai_summarize_agg_tool)

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
