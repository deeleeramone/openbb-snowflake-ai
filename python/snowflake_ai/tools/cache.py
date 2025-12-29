"""SQL query caching utilities for tool execution."""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from .helpers import _split_identifier

if TYPE_CHECKING:
    from .._snowflake_ai import SnowflakeAI

from ..logger import get_logger

logger = get_logger(__name__)

SQL_CACHE_KEY = "text2sql_cache_v1"
SQL_CACHE_MAX_ENTRIES = 5
SQL_CACHE_ROW_LIMIT = 500

SQL_TABLE_PATTERN = re.compile(
    r"\b(?:from|join|into|update|table)\s+([\w\.\"$]+)",
    re.IGNORECASE,
)


async def _load_sql_cache(client: "SnowflakeAI", conv_id: str) -> dict:
    """Load SQL query cache from conversation data."""
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


async def _persist_sql_cache(client: "SnowflakeAI", conv_id: str, cache: dict):
    """Persist SQL query cache to conversation data."""
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
    """Shrink cache to max entries, keeping most recent."""
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


def _extract_tables_from_sql(
    sql: str, default_db: str | None, default_schema: str | None
) -> list[dict[str, str]]:
    """Extract table references from SQL query."""
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
    client: "SnowflakeAI", tables: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Capture last_altered timestamps for cache invalidation."""
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


async def _is_cache_entry_stale(client: "SnowflakeAI", entry: dict) -> bool:
    """Check if a cache entry is stale based on table freshness."""
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
    client: "SnowflakeAI",
    conv_id: str,
    cache: dict,
    query_hash: str,
    query: str,
    result_json: dict,
    row_count: int,
    tables: list[dict[str, str]],
):
    """Cache a query result with table freshness metadata."""
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
