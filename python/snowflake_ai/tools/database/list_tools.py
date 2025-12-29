"""Database list tool handlers."""

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_list_databases(ctx: "ToolContext", args: dict):
    """Handle list_databases tool call."""
    try:
        result = await asyncio.to_thread(ctx.client.list_databases)
        output = "Available databases:\n" + "\n".join(f"- {db}" for db in result)
        yield output, result
    except Exception as e:
        error_msg = f"Error listing databases: {str(e)}"
        yield error_msg, {"error": str(e)}


async def handle_list_schemas(ctx: "ToolContext", args: dict):
    """Handle list_schemas tool call."""
    database = args.get("database")
    try:
        result = await asyncio.to_thread(ctx.client.list_schemas, database)
        db_name = database or "current database"
        output = f"Schemas in {db_name}:\n" + "\n".join(f"- {s}" for s in result)
        yield output, result
    except Exception as e:
        error_msg = f"Error listing schemas: {str(e)}"
        yield error_msg, {"error": str(e)}


async def handle_list_tables_in(ctx: "ToolContext", args: dict):
    """Handle list_tables_in tool call."""
    database = args.get("database", "")
    schema = args.get("schema", "")
    try:
        result = await asyncio.to_thread(ctx.client.list_tables_in, database, schema)
        output = f"Tables in {database}.{schema}:\n" + "\n".join(
            f"- {t}" for t in result
        )
        yield output, result
    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        yield error_msg, {"error": str(e)}
