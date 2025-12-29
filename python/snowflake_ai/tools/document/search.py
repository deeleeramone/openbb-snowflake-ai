"""Document search tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

from openbb_ai import reasoning_step

from ...document_processor import DocumentProcessor
from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_search_document(ctx: "ToolContext", args: dict):
    """Handle search_document tool call."""
    query = args.get("query", "")
    file_name = args.get("file_name")
    top_k = args.get("top_k", 5)

    if not query:
        yield "Error: query is required", {"error": "query required"}
        return

    try:
        doc_proc = DocumentProcessor.instance()
        conv_id = args.get("_conversation_id", ctx.conv_id)

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
                snowflake_user = await asyncio.to_thread(ctx.client.get_current_user)
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

                result = await asyncio.to_thread(ctx.client.execute_query, table_sql)
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
                        page_num = row.get("PAGE_NUMBER", row.get("page_number", "?"))
                        fname = row.get("FILE_NAME", row.get("file_name", "Unknown"))

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
            ctx.client,
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
                snowflake_user = await asyncio.to_thread(ctx.client.get_current_user)
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

                result = await asyncio.to_thread(ctx.client.execute_query, keyword_sql)
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
                output += " | 🖼️ **IMAGE**\n"
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

    except Exception as e:
        error_msg = f"Error searching documents: {str(e)}"
        yield error_msg, {"error": str(e)}
