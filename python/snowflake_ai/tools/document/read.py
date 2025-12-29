"""Read document tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

from openbb_ai.helpers import table

from ...helpers import extract_markdown_tables, to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


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

        # Format document content and extract tables
        output = f"**Document: {file_name}** ({len(row_data)} pages)\n\n"
        all_extracted_tables = []

        for row in row_data:
            page_num = row.get("PAGE_NUMBER", row.get("page_number", "?"))
            content = row.get("PAGE_CONTENT", row.get("page_content", ""))

            # Extract any markdown tables from the page content
            logger.info(
                "Page %s content length: %d, has pipe chars: %s",
                page_num,
                len(content),
                "|" in content,
            )
            modified_content, extracted_tables = extract_markdown_tables(content)
            logger.info("Page %s: extracted %d tables", page_num, len(extracted_tables))

            # Emit table artifacts for any extracted tables
            for tbl in extracted_tables:
                tbl["name"] = f"Page {page_num} - {tbl['name']}"
                all_extracted_tables.append(tbl)
                logger.info(
                    "Creating table artifact: %s with %d rows",
                    tbl["name"],
                    len(tbl["data"]),
                )
                try:
                    table_artifact = table(
                        data=tbl["data"],
                        name=tbl["name"],
                        description=f"Table from {file_name}, page {page_num}",
                    )
                    sse_event = to_sse(table_artifact)
                    logger.info(
                        "Yielding table SSE event: %s",
                        str(sse_event)[:200],
                    )
                    yield sse_event
                except Exception as table_err:
                    logger.warning("Failed to create table artifact: %s", table_err)

            # Add page content (with table placeholders replaced by descriptions)
            if extracted_tables:
                # Replace placeholders with clear artifact references
                for tbl in extracted_tables:
                    headers = tbl.get(
                        "headers", list(tbl["data"][0].keys()) if tbl["data"] else []
                    )
                    artifact_note = (
                        f"📊 **[TABLE ARTIFACT: {tbl['name']}]**\n"
                        f"   Columns: {', '.join(headers)} | Rows: {len(tbl['data'])}\n"
                        f"   *(Table displayed as interactive artifact - do not re-output)*"
                    )
                    modified_content = modified_content.replace(
                        tbl["placeholder"], artifact_note
                    )
                output += f"---\n**Page {page_num}:**\n{modified_content}\n\n"
            else:
                output += f"---\n**Page {page_num}:**\n{content}\n\n"

        yield output, {
            "file_name": file_name,
            "pages": len(row_data),
            "content": row_data,
            "tables_extracted": len(all_extracted_tables),
        }

    except Exception as e:
        error_msg = f"Error reading document '{file_name}': {str(e)}"
        yield error_msg, {"error": str(e)}
