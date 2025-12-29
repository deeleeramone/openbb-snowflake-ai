"""Document images tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_get_document_images(ctx: "ToolContext", args: dict):
    """Handle get_document_images tool call."""
    file_name = args.get("file_name", "")
    page_numbers = args.get("page_numbers", [])

    if not file_name:
        yield "Error: file_name is required", {"error": "file_name required"}
        return

    try:
        # Get user schema
        snowflake_user = await asyncio.to_thread(ctx.client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        user_schema = f"USER_{sanitized_user}".upper()

        file_name_escaped = file_name.replace("'", "''")

        # Build page filter if provided
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

        result = await asyncio.to_thread(ctx.client.execute_statement, query_sql)

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
            result = await asyncio.to_thread(ctx.client.execute_statement, fallback_sql)
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
                    "embedding_id": row.get("EMBEDDING_ID") or row.get("embedding_id"),
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
                    "image_format": row.get("IMAGE_FORMAT") or row.get("image_format"),
                    "width": row.get("WIDTH") or row.get("width"),
                    "height": row.get("HEIGHT") or row.get("height"),
                    "has_embedding": row.get("HAS_EMBEDDING")
                    or row.get("has_embedding"),
                }

            images.append(img_data)

            output += "---\n"
            output += (
                f"**Image {img_data['image_index']}** | Page {img_data['page_number']}"
            )
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

    except Exception as e:
        error_msg = f"Error getting document images: {str(e)}"
        yield error_msg, {"error": str(e)}
