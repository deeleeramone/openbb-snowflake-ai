"""Tool execution for Snowflake AI.

This module is a thin facade that re-exports the tool execution functions
from the tools package for backward compatibility.

The actual tool handlers have been moved to individual files under tools/:
  - tools/database/: list_databases, list_schemas, list_tables_in, etc.
  - tools/query/: text2sql, execute_query, execute_statement, validate_query
  - tools/document/: read_document, search_document, get_document_images, ocr_image
  - tools/ai_functions/: ai_filter, ai_agg, ai_summarize_agg, extract_answer
  - tools/cortex/: sentiment, summarize, translate
  - tools/charts/: render_chart
  - tools/pagination/: continue_output
"""

from .tools import (
    execute_tool,
    get_tool_definitions,
    ToolState,
    get_last_query_results,
    set_last_query_result,
    TOOL_REGISTRY,
)

# Backward compatibility: streaming_handler.py accesses tool_executor._last_query_results
# We expose the ToolState's last_query_results dict at module level.
# This is the same dict instance, so mutations are shared.
_last_query_results = ToolState.instance().last_query_results

__all__ = [
    "execute_tool",
    "get_tool_definitions",
    "_last_query_results",
    "TOOL_REGISTRY",
    "ToolState",
    "get_last_query_results",
    "set_last_query_result",
]
