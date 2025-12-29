"""Read document tool handler."""

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext

logger = logging.getLogger(__name__)


async def handle_read_document(ctx: "ToolContext", args: dict):
    """Handle read_document tool call."""
    file_name = args.get("file_name", "")
    page_numbers = args.get("page_numbers", [])

    logger.info(
        "read_document called: file_name=%s, page_numbers=%s", file_name, page_numbers
    )

    if not file_name:
        yield "Error: file_name is required", {"error": "file_name required"}
        return

    try:
        # Get user schema
        logger.debug("Getting current user...")
        snowflake_user = await asyncio.to_thread(ctx.client.get_current_user)
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

        result = await asyncio.to_thread(ctx.client.execute_query, query)
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

    except Exception as e:
        error_msg = f"Error reading document '{file_name}': {str(e)}"
        yield error_msg, {"error": str(e)}
