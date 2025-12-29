"""Chart rendering tool handler."""

from typing import TYPE_CHECKING

from openbb_ai import reasoning_step
from openbb_ai.helpers import chart

from ..base import get_last_query_results
from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_render_chart(ctx: "ToolContext", args: dict):
    """Handle render_chart tool call."""
    chart_type = args.get("chart_type", "bar").lower()
    title = args.get("title", "Chart")
    data = args.get("data", [])
    label_column = args.get("label_column", "")
    value_column = args.get("value_column", "")
    conv_id = args.get("_conversation_id", ctx.conv_id)

    # If no data provided but we have cached query results, use those
    if not data:
        cached_results = get_last_query_results()
        if conv_id in cached_results:
            data = cached_results[conv_id]
            yield to_sse(
                reasoning_step(
                    f"Using cached query results ({len(data)} rows) for chart",
                    event_type="INFO",
                )
            )

    # Validate inputs
    if not data:
        error_msg = "Error: No data available for chart. Please run a query first using execute_query."
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Missing data"}
        return

    if not label_column or not value_column:
        # Try to auto-detect columns from data
        if data and isinstance(data[0], dict):
            columns = list(data[0].keys())
            # Heuristic: first text column is label, first numeric column is value
            for col in columns:
                sample_val = data[0].get(col)
                if not label_column and isinstance(sample_val, str):
                    label_column = col
                elif not value_column and isinstance(sample_val, (int, float)):
                    value_column = col

        if not label_column or not value_column:
            error_msg = f"Error: 'label_column' and 'value_column' are required. Available columns: {list(data[0].keys()) if data else []}"
            yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
            yield error_msg, {"error": "Missing column names"}
            return

    # Extract labels and values from the data array
    labels = []
    values = []
    for row in data:
        if isinstance(row, dict):
            # Try both exact case and uppercase (Snowflake returns uppercase)
            label_val = (
                row.get(label_column)
                or row.get(label_column.upper())
                or row.get(label_column.lower())
            )
            value_val = (
                row.get(value_column)
                or row.get(value_column.upper())
                or row.get(value_column.lower())
            )

            if label_val is not None:
                labels.append(str(label_val))
            if value_val is not None:
                try:
                    values.append(float(value_val))
                except (ValueError, TypeError):
                    values.append(0)

    if not labels or not values:
        error_msg = f"Error: Could not extract data from columns '{label_column}' and '{value_column}'. Available: {list(data[0].keys()) if data else []}"
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Column extraction failed"}
        return

    if len(labels) != len(values):
        error_msg = (
            f"Error: labels ({len(labels)}) and values ({len(values)}) count mismatch"
        )
        yield to_sse(reasoning_step(error_msg, event_type="ERROR"))
        yield error_msg, {"error": "Length mismatch"}
        return

    # Map chart types
    type_map = {
        "pie": "pie",
        "donut": "donut",
        "bar": "bar",
        "line": "line",
        "scatter": "scatter",
        "area": "area",
    }
    openbb_chart_type = type_map.get(chart_type, "bar")

    # Build chart data as list of dicts (the format openbb_ai.chart expects)
    chart_data = []
    for label, value in zip(labels, values):
        chart_data.append(
            {
                "label": label,
                "value": value,
            }
        )

    try:
        if openbb_chart_type in ("pie", "donut"):
            chart_artifact = chart(
                type=openbb_chart_type,
                data=chart_data,
                angle_key="value",
                callout_label_key="label",
                name=title,
                description=f"{chart_type.capitalize()} chart with {len(chart_data)} data points",
            )
        else:
            chart_artifact = chart(
                type=openbb_chart_type,  # type: ignore
                data=chart_data,
                x_key="label",
                y_keys=["value"],
                name=title,
                description=f"{chart_type.capitalize()} chart with {len(chart_data)} data points",
            )

        yield to_sse(chart_artifact)
        yield f"✅ Rendered {chart_type} chart: {title}", {
            "chart_type": chart_type,
            "title": title,
            "data_points": len(labels),
        }

    except Exception as chart_error:
        logger.error("Failed to create chart: %s", chart_error)
        yield to_sse(
            reasoning_step(f"Chart creation failed: {chart_error}", event_type="ERROR")
        )
        yield f"Error creating chart: {chart_error}", {"error": str(chart_error)}
