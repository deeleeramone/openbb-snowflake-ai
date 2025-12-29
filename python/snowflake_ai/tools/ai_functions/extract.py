"""Extract answer tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_extract_answer(ctx: "ToolContext", args: dict):
    """Handle extract_answer tool call."""
    # Use AI_EXTRACT SQL function instead of deprecated CLI command
    file_path = args.get("file_path", "")
    questions = args.get("questions", [])

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

        result = await asyncio.to_thread(ctx.client.execute_query, query)
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

    except Exception as e:
        error_msg = f"Error executing AI_EXTRACT: {e}"
        yield error_msg, {"error": str(e)}
