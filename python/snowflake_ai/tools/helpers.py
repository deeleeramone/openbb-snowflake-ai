"""Helper functions for tool execution."""

import asyncio
import hashlib
import re
import shutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._snowflake_ai import SnowflakeAI


def _normalize_sql(sql: str) -> str:
    """Normalize SQL by collapsing whitespace."""
    return re.sub(r"\s+", " ", sql or "").strip()


def _hash_sql(sql: str) -> str:
    """Create SHA256 hash of SQL query."""
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _split_identifier(identifier: str) -> list[str]:
    """Split a potentially quoted Snowflake identifier into parts."""
    parts: list[str] = []
    current: list[str] = []
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


async def _get_context_defaults(client: "SnowflakeAI") -> tuple[str | None, str | None]:
    """Get current database and schema from client context."""
    try:
        database = await asyncio.to_thread(client.get_current_database)
    except Exception:  # pragma: no cover - defensive
        database = None
    try:
        schema = await asyncio.to_thread(client.get_current_schema)
    except Exception:  # pragma: no cover - defensive
        schema = None
    return database, schema


async def _get_database_context_for_llm(client: "SnowflakeAI", question: str) -> str:
    """Build database context string for LLM SQL generation fallback.

    Fetches table schemas and builds a context string that helps the LLM
    generate valid Snowflake SQL queries.
    """
    try:
        database, schema = await _get_context_defaults(client)

        if not database or not schema:
            return "No database context available."

        # Get list of tables in current schema
        tables = await asyncio.to_thread(client.list_tables_in, database, schema)

        if not tables:
            return f"Database: {database}, Schema: {schema}\nNo tables found."

        context_lines = [
            f"Current database: {database}",
            f"Current schema: {schema}",
            "",
            "Available tables and their columns:",
        ]

        # Get schema for ALL tables - DO NOT TRUNCATE
        # The LLM needs complete context to make correct decisions
        import json

        for table_name in tables:
            try:
                table_info_json = await asyncio.to_thread(
                    client.get_table_info, table_name
                )
                table_info = json.loads(table_info_json)
                columns = table_info.get("columns", [])

                if columns:
                    col_strs = []
                    for col in columns:  # ALL columns, not truncated
                        col_name = col.get("COLUMN_NAME", "?")
                        col_type = col.get("DATA_TYPE", "?")
                        col_strs.append(f"{col_name} ({col_type})")

                    fq_name = f"{database}.{schema}.{table_name}"
                    context_lines.append(f"  {fq_name}: {', '.join(col_strs)}")
            except Exception:
                continue

        return "\n".join(context_lines)
    except Exception as e:
        return f"Error fetching database context: {e}"


def _find_snow_cli_binary() -> str | None:
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
