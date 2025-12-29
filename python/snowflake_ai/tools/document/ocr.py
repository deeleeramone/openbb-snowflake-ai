"""OCR image tool handler."""

import asyncio
import json
import re
from typing import TYPE_CHECKING

from openbb_ai import reasoning_step

from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_ocr_image(ctx: "ToolContext", args: dict):
    """Handle ocr_image tool call."""
    image_stage_path = args.get("image_stage_path")
    file_name = args.get("file_name")
    page_number = args.get("page_number")
    # extract_tables is accepted but vision model handles this automatically

    try:
        # Get user schema
        snowflake_user = await asyncio.to_thread(ctx.client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        user_schema = f"USER_{sanitized_user}".upper()

        # If file_name and page_number provided, look up the image path
        # Try DOCUMENT_IMAGES_METADATA first (rendered page images), then DOCUMENT_EMBEDDINGS
        if not image_stage_path and file_name and page_number:
            file_name_escaped = file_name.replace("'", "''")
            logger.debug(
                "[ocr_image] Looking up image for file=%s, page=%s",
                file_name,
                page_number,
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
                result = await asyncio.to_thread(ctx.client.execute_query, metadata_sql)
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
                        logger.debug(
                            "[ocr_image] Found in DOCUMENT_IMAGES_METADATA: %s",
                            image_stage_path,
                        )
                    else:
                        logger.debug(
                            "[ocr_image] DOCUMENT_IMAGES_METADATA returned no rows"
                        )
            except Exception as meta_err:
                logger.warning(
                    "[ocr_image] DOCUMENT_IMAGES_METADATA lookup failed: %s", meta_err
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
                    ctx.client.execute_query, embeddings_sql
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
                        ctx.client.execute_statement, list_sql
                    )
                    if result:
                        rows = json.loads(result) if isinstance(result, str) else result
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
                                logger.debug(
                                    "Found image via stage LIST: %s", image_stage_path
                                )
                                break
                except Exception as list_err:
                    logger.warning("Stage LIST fallback failed: %s", list_err)

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
        return_as_chart = args.get("return_as_chart", False)

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
            logger.debug(
                "[ocr_image] Calling vision model with SQL: %s...", vision_sql[:200]
            )
            result = await asyncio.to_thread(ctx.client.execute_statement, vision_sql)
            logger.debug(
                "[ocr_image] Vision model returned: %s",
                str(result)[:500] if result else "None",
            )
        except Exception as vision_exc:
            vision_error = str(vision_exc)
            logger.error("[ocr_image] Vision model failed: %s", vision_error)

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
                    vision_result = row.get("VISION_RESULT") or row.get("vision_result")

                # Parse the vision result (should be JSON)
                if isinstance(vision_result, str):
                    # Try to extract JSON from the response (model may add explanation text)
                    try:
                        # Try direct parse first
                        extracted_data = json.loads(vision_result)
                    except json.JSONDecodeError:
                        # Try to find JSON in the response
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
                logger.debug("[ocr_image] Calling OCR with SQL: %s...", ocr_sql[:200])
                ocr_result = await asyncio.to_thread(
                    ctx.client.execute_statement, ocr_sql
                )
                logger.debug(
                    "[ocr_image] OCR returned: %s",
                    str(ocr_result)[:500] if ocr_result else "None",
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
                logger.error("[ocr_image] OCR fallback also failed: %s", ocr_exc)
                extracted_text = f"Unable to extract content from image. Vision error: {vision_error or 'no result'}. OCR error: {ocr_exc}"

        # ALWAYS return something - never exit without output
        if not extracted_data and not extracted_text:
            extracted_text = f"Image found at {image_stage_path} but could not extract content. Vision error: {vision_error or 'unknown'}. Please try read_document_page to get the text content of this page."
            logger.warning(
                "[ocr_image] No extraction succeeded, returning fallback message"
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

            logger.debug(
                "[ocr_image] Parsed %d data points from %d section(s)",
                len(data_points),
                len(sections) or 1,
            )
            logger.debug(
                "[ocr_image] Parsed %d data points from %d sections: %s",
                len(data_points),
                len(sections),
                sections,
            )
            logger.debug(
                "[ocr_image] Data points sample: %s%s",
                data_points[:3],
                "..." if len(data_points) > 3 else "",
            )

            if chart_type and data_points and return_as_chart:
                # Return as OpenBB chart artifact
                try:
                    from openbb_ai.helpers import chart

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

                    chart_title = extracted_data.get("title") or "Extracted Chart Data"
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

                        logger.debug(
                            "[ocr_image] Creating charts for %d sections: %s",
                            len(section_data),
                            list(section_data.keys()),
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

                                logger.debug(
                                    "[ocr_image] Creating chart '%s' with %d points",
                                    display_name,
                                    len(chart_points),
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
                        summary_text += (
                            f"**Data points:** {len(chart_data)} values extracted\n\n"
                        )
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
                    logger.warning("openbb_ai.helpers not available for chart output")
                except Exception as chart_exc:
                    logger.warning("Failed to create chart artifact: %s", chart_exc)

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

    except Exception as e:
        error_msg = f"Error performing OCR: {str(e)}"
        logger.error(error_msg, exc_info=True)
        yield error_msg, {"error": str(e)}
