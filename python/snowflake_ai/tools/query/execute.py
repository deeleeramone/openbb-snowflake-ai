"""Query execution tool handlers."""

import asyncio
import json
import os
from typing import TYPE_CHECKING

from openbb_ai.helpers import table
from ..base import set_last_query_result
from ..cache import (
    _load_sql_cache,
    _is_cache_entry_stale,
    _extract_tables_from_sql,
    _cache_query_result,
)
from ..helpers import _normalize_sql, _hash_sql, _get_context_defaults
from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_execute_query(ctx: "ToolContext", args: dict):
    """Handle execute_query tool call."""
    query = args.get("query", "")
    conv_id = args.get("_conversation_id", ctx.conv_id)
    force_refresh = bool(args.get("force_refresh"))

    # Use a per-request UUID to make artifact names unique and avoid UI deduping
    import uuid as _uuid

    uuid_str = str(_uuid.uuid4())

    normalized_query = _normalize_sql(query)
    query_hash = _hash_sql(normalized_query)
    cache = await _load_sql_cache(ctx.client, conv_id)

    # TEMPORARILY DISABLED: Cache is causing stale artifacts to be yielded
    # TODO: Debug why cache is returning wrong data
    use_cache = False
    if use_cache and not force_refresh and query_hash in cache:
        cache_entry = cache[query_hash]
        try:
            is_stale = await _is_cache_entry_stale(ctx.client, cache_entry)
        except Exception as freshness_error:  # pragma: no cover - defensive
            logger.debug("Cache freshness check failed: %s", freshness_error)
            is_stale = True

        if not is_stale:
            cached_result = cache_entry.get("result")
            if isinstance(cached_result, dict):
                reuse_msg = cache_entry.get("executed_at", "cached result")
                row_data = cached_result.get("rowData", [])
                num_rows = len(row_data)

                # Store the last query result for charting
                set_last_query_result(conv_id, row_data)

                # Generate table artifact for cached data too
                if num_rows > 0 and row_data:
                    try:
                        unique_name = f"Query Results ({uuid_str[:8]})"
                        table_artifact = table(
                            data=row_data,
                            name=unique_name,
                            description=(
                                query[:200]
                                if len(query) <= 200
                                else query[:197] + "..."
                            ),
                        )
                        logger.debug(
                            "[ARTIFACT NAME] Emitting cached table artifact name=%s rows=%d",
                            unique_name,
                            num_rows,
                        )
                        yield to_sse(table_artifact)
                    except Exception as artifact_error:
                        logger.warning(
                            "Failed to create table artifact from cache: %s",
                            artifact_error,
                        )

                # Pass the FULL data to the LLM as JSON
                if num_rows == 0:
                    output = f"Reusing cached result from {reuse_msg}. Query returned 0 rows."
                else:
                    output = (
                        f"Reusing cached result from {reuse_msg}. "
                        f"Query returned {num_rows} row(s). Here is the complete result data:\n\n"
                        f"```json\n{json.dumps(row_data, indent=2, default=str)}\n```\n\n"
                        "⚠️ A TABLE ARTIFACT HAS ALREADY BEEN DISPLAYED TO THE USER. "
                        "DO NOT repeat this data as a markdown table. "
                        "Instead, provide insights, analysis, or a summary of the key findings."
                    )

                yield output, cached_result
                return

    try:
        result = await asyncio.to_thread(ctx.client.execute_query, query)
        result_json = json.loads(result)
        row_data = result_json.get("rowData", [])
        num_rows = len(row_data)

        # DEBUG: Log what we're actually getting
        logger.debug(
            "[DEBUG] execute_query received %d rows for query: %s...",
            num_rows,
            query[:100],
        )
        if row_data:
            logger.debug(
                "[DEBUG] First row keys: %s",
                list(row_data[0].keys()) if row_data else "N/A",
            )

        # Store the last query result for use by render_chart
        set_last_query_result(conv_id, row_data)

        # Generate table artifact if we have data
        if num_rows > 0 and row_data:
            try:
                from openbb_ai.helpers import table

                # CRITICAL DEBUG: Log exactly what we're putting in the artifact
                first_row_preview = str(row_data[0])[:200] if row_data else "EMPTY"
                logger.debug(
                    "[ARTIFACT DEBUG] Creating table with %d rows. First row: %s",
                    len(row_data),
                    first_row_preview,
                )
                logger.debug("[ARTIFACT DEBUG] Query was: %s...", query[:150])

                # Create table artifact with query results
                unique_name = f"Query Results ({uuid_str[:8]})"
                table_artifact = table(
                    data=row_data,
                    name=unique_name,
                    description=(
                        query[:200] if len(query) <= 200 else query[:197] + "..."
                    ),
                )
                logger.debug(
                    "[ARTIFACT NAME] Emitting fresh table artifact name=%s rows=%d",
                    unique_name,
                    len(row_data),
                )

                # Log the artifact UUID so we can trace it
                artifact_uuid = (
                    str(table_artifact.data.uuid)
                    if hasattr(table_artifact, "data")
                    and hasattr(table_artifact.data, "uuid")
                    else "UNKNOWN"
                )
                logger.debug("[ARTIFACT DEBUG] Artifact UUID: %s", artifact_uuid)
                logger.debug(
                    "[ARTIFACT DEBUG] Artifact created with %d rows, yielding now",
                    len(row_data),
                )

                # Convert to SSE and log what we're actually yielding
                sse_event = to_sse(table_artifact)
                logger.debug(
                    "[ARTIFACT DEBUG] SSE event type: %s", sse_event.get("event")
                )
                # Log first 500 chars of the data to verify content
                data_preview = sse_event.get("data", "")[:500]
                logger.debug("[ARTIFACT DEBUG] SSE data preview: %s", data_preview)

                # Yield table artifact for UI
                yield sse_event

            except Exception as artifact_error:
                logger.warning("Failed to create table artifact: %s", artifact_error)

        # Generate text output for LLM - INCLUDE THE FULL DATA so the LLM can answer questions
        if num_rows == 0:
            output = "Query executed successfully. Returned 0 rows."
        else:
            # Pass the FULL data to the LLM as JSON so it can answer accurately
            # The table artifact handles UI display; this is for the LLM to use
            output = (
                f"Query returned {num_rows} row(s). Here is the complete result data:\n\n"
                f"```json\n{json.dumps(row_data, indent=2, default=str)}\n```\n\n"
                "⚠️ A TABLE ARTIFACT HAS ALREADY BEEN DISPLAYED TO THE USER. "
                "DO NOT repeat this data as a markdown table. "
                "Instead, provide insights, analysis, or a summary of the key findings."
            )

        yield output, result_json

        try:
            default_db, default_schema = await _get_context_defaults(ctx.client)
            tables = _extract_tables_from_sql(
                normalized_query, default_db, default_schema
            )
            await _cache_query_result(
                ctx.client,
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

    except Exception as e:
        error_msg = f"Query execution failed: {str(e)}"
        logger.error("Query execution error: %s", e)
        yield error_msg, {"error": str(e)}


async def handle_execute_statement(ctx: "ToolContext", args: dict):
    """Handle execute_statement tool call."""
    statement = (args.get("statement", "") or "").strip()
    if not statement:
        yield "Error: 'statement' argument is required.", {"error": "Missing statement"}
        return

    try:
        result = await asyncio.to_thread(ctx.client.execute_statement, statement)
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

    except Exception as e:
        error_msg = f"Statement execution failed: {str(e)}"
        logger.error("Statement execution error: %s", e)
        yield error_msg, {"error": str(e)}
