"""AI Chat Tools for Snowflake.

This module provides a registry of all available tools and the main
execute_tool and get_tool_definitions functions.
"""

import json
from typing import TYPE_CHECKING, Any

from openbb_ai import reasoning_step

from .base import ToolContext, ToolState, get_last_query_results, set_last_query_result
from ..helpers import to_sse
from ..logger import get_logger

# Import all tool handlers
from .database import (
    handle_list_databases,
    handle_list_schemas,
    handle_list_tables_in,
    handle_get_table_sample_data,
    handle_get_table_schema,
    handle_get_multiple_table_definitions,
    handle_list_semantic_views,
)
from .query import (
    handle_text2sql,
    handle_execute_query,
    handle_execute_statement,
    handle_validate_query,
)
from .document import (
    handle_read_document,
    handle_search_document,
    handle_get_document_images,
    handle_ocr_image,
)
from .ai_functions import (
    handle_ai_filter,
    handle_ai_agg,
    handle_ai_summarize_agg,
    handle_extract_answer,
)
from .cortex import (
    handle_sentiment,
    handle_summarize,
    handle_translate,
)
from .charts import handle_render_chart
from .pagination import handle_continue_output

if TYPE_CHECKING:
    from .._snowflake_ai import SnowflakeAI, ToolCall

logger = get_logger(__name__)

# Tool registry mapping tool names to handler functions
TOOL_REGISTRY = {
    # Database tools
    "list_databases": handle_list_databases,
    "list_schemas": handle_list_schemas,
    "list_tables_in": handle_list_tables_in,
    "get_table_sample_data": handle_get_table_sample_data,
    "get_table_schema": handle_get_table_schema,
    "get_multiple_table_definitions": handle_get_multiple_table_definitions,
    "list_semantic_views": handle_list_semantic_views,
    # Query tools
    "text2sql": handle_text2sql,
    "execute_query": handle_execute_query,
    "execute_statement": handle_execute_statement,
    "validate_query": handle_validate_query,
    # Document tools
    "read_document": handle_read_document,
    "search_document": handle_search_document,
    "get_document_images": handle_get_document_images,
    "ocr_image": handle_ocr_image,
    # AI function tools
    "ai_filter": handle_ai_filter,
    "ai_agg": handle_ai_agg,
    "ai_summarize_agg": handle_ai_summarize_agg,
    "extract_answer": handle_extract_answer,
    # Cortex NLP tools
    "sentiment": handle_sentiment,
    "summarize": handle_summarize,
    "translate": handle_translate,
    # Charts
    "render_chart": handle_render_chart,
    # Pagination
    "continue_output": handle_continue_output,
}

# Tools that provide their own reasoning steps
TOOLS_WITH_OWN_REASONING = {"text2sql"}


async def execute_tool(
    tool_call: "ToolCall",
    client: "SnowflakeAI",
    conv_id: str = "default",
):
    """
    Execute a tool call and yield reasoning steps and final output.

    Parameters
    ----------
    tool_call : ToolCall
        The tool call to execute
    client : SnowflakeAI
        The Snowflake client
    conv_id : str
        Conversation ID for accessing cached metadata

    Yields:
        Reasoning steps and a final Tuple of (llm_output, raw_data)
    """
    tool_name = tool_call.function.name
    tool_args_str = tool_call.function.arguments

    try:
        tool_args = json.loads(tool_args_str) if tool_args_str else {}
    except json.JSONDecodeError:
        tool_args = {"raw": tool_args_str}

    # Inject conversation ID for tools that need cached metadata
    tool_args["_conversation_id"] = conv_id

    # Don't yield "Executing tool" here - the server already yields "Calling tool"
    # which is sufficient. We only yield tool-specific reasoning from handlers.

    # Look up handler in registry
    handler = TOOL_REGISTRY.get(tool_name)
    if handler:
        ctx = ToolContext(client=client, conv_id=conv_id)
        async for result in handler(ctx, tool_args):
            yield result
    else:
        # Unknown tool
        error_msg = f"Unknown tool: {tool_name}"
        yield error_msg, {"error": error_msg}


def get_tool_definitions(client: "SnowflakeAI") -> list:
    """Get the list of available tool definitions."""
    raw_tools = [
        client.text2sql_tool(),
        client.get_table_sample_data_tool(),
        client.get_table_schema_tool(),
        client.get_multiple_table_definitions_tool(),
        client.list_databases_tool(),
        client.list_schemas_tool(),
        client.list_tables_in_tool(),
        client.list_semantic_views_tool(),
        client.validate_query_tool(),
        client.execute_query_tool(),
        client.execute_statement_tool(),
        client.extract_answer_tool(),
        client.sentiment_tool(),
        client.summarize_tool(),
        client.translate_tool(),
    ]

    # Add get_widget_data tool for fetching dashboard widget data
    get_widget_data_tool = {
        "type": "function",
        "function": {
            "name": "get_widget_data",
            "description": "Fetch data from dashboard widgets. Use this tool when the user asks about charts, tables, or other visualizations on their dashboard. The data will be retrieved from the widget and made available for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "widget_uuids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of widget UUIDs to fetch data from. If not specified, data from all available widgets will be fetched.",
                    }
                },
                "required": [],
            },
        },
    }
    raw_tools.append(get_widget_data_tool)

    # Add read_document tool for easy document content retrieval
    read_document_tool = {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read specific pages from an uploaded document. IMPORTANT: Always specify page_numbers to avoid context overflow! Use search_document first to find relevant pages, then read those specific pages. Maximum recommended: 5-10 pages per call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The exact filename of the document to read (e.g., 'technology-investment.pdf'). Match user's description to the available documents list.",
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "RECOMMENDED: List of specific page numbers to retrieve (e.g., [1, 5, 12]). Use search_document first to find relevant pages. Omitting this parameter loads ALL pages which may cause context overflow on large documents!",
                    },
                },
                "required": ["file_name"],
            },
        },
    }
    raw_tools.append(read_document_tool)

    # Add search_document tool for semantic search across documents
    search_document_tool = {
        "type": "function",
        "function": {
            "name": "search_document",
            "description": "Search uploaded documents using semantic similarity. Use this to find specific information, topics, or answers within documents. Returns the most relevant text chunks with similarity scores. More efficient than reading entire documents when looking for specific content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query - describe what you're looking for in natural language (e.g., 'revenue growth projections', 'risk factors mentioned', 'CEO statement on market outlook').",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Optional: limit search to a specific document filename. If not provided, searches across all uploaded documents.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top matching chunks to return (default: 5, max: 20).",
                    },
                },
                "required": ["query"],
            },
        },
    }
    raw_tools.append(search_document_tool)

    # Add get_document_images tool for retrieving images from documents
    get_document_images_tool = {
        "type": "function",
        "function": {
            "name": "get_document_images",
            "description": "Get images from a document by page number. Returns image URLs that can be displayed or analyzed. Use after search_document finds relevant image results, or to get all images from specific pages. Images include charts, graphs, diagrams, and photos extracted from the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_name": {
                        "type": "string",
                        "description": "The document filename to get images from.",
                    },
                    "page_numbers": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of page numbers to get images from (e.g., [1, 5, 10]). If not provided, returns all images.",
                    },
                },
                "required": ["file_name"],
            },
        },
    }
    raw_tools.append(get_document_images_tool)

    # Add ocr_image tool for extracting text from images
    # Uses vision model (claude-3-5-sonnet) for accurate chart/graph extraction
    ocr_image_tool = {
        "type": "function",
        "function": {
            "name": "ocr_image",
            "description": "Extract text and data from an image using AI vision. BEST FOR: reading charts, graphs, and visualizations where spatial relationships matter (correctly matching X-axis labels to values). Also works for tables, diagrams, and general images. Uses vision model for accurate chart extraction, with OCR fallback. Use after get_document_images to analyze specific images.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_stage_path": {
                        "type": "string",
                        "description": "The stage path to the image (e.g., '@OPENBB_AGENTS.USER_DLEE.DOCUMENT_IMAGES/doc.pdf/page_5_image_0.jpeg'). Get this from get_document_images results.",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Alternative: document filename. If provided with page_number, will analyze that page's image.",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Page number to analyze (use with file_name instead of image_stage_path).",
                    },
                    "extract_tables": {
                        "type": "boolean",
                        "description": "If true, attempt to extract tabular data as structured format. Default: true.",
                    },
                    "return_as_chart": {
                        "type": "boolean",
                        "description": "If true and image contains a chart, return as interactive chart artifact. Chart type is determined by the extracted content. Default: false.",
                    },
                },
                "required": [],
            },
        },
    }
    raw_tools.append(ocr_image_tool)

    # Add ai_filter tool for boolean classification of text/data using AI
    ai_filter_tool = {
        "type": "function",
        "function": {
            "name": "ai_filter",
            "description": "Use AI to classify text/data as TRUE or FALSE based on a natural language condition. Best for: (1) Filtering query results by semantic meaning - e.g., filter reviews where 'customer sounds satisfied', (2) Yes/no classification of text - e.g., 'Is this about financial risk?', (3) Image classification - e.g., 'Is this a product photo?'. NOT for: extracting data (use extract_answer), summarizing (use summarize), or searching documents (use search_document).",
            "parameters": {
                "type": "object",
                "properties": {
                    "predicate": {
                        "type": "string",
                        "description": "Natural language condition to evaluate as TRUE/FALSE. Phrase as statement: 'The customer sounds satisfied', 'This discusses financial risk', 'The sentiment is positive'. Be specific for accuracy.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to classify. Use for single text classification without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query whose results will be filtered. AI_FILTER evaluates each row against the predicate.",
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Column name to apply AI filter to when using 'query'. Required with query parameter.",
                    },
                    "image_stage_path": {
                        "type": "string",
                        "description": "Stage path to image for classification (e.g., '@DB.SCHEMA.STAGE/img.jpg'). Use instead of text for images.",
                    },
                },
                "required": ["predicate"],
            },
        },
    }
    raw_tools.append(ai_filter_tool)

    # Add ai_agg tool for AI-powered text aggregation
    ai_agg_tool = {
        "type": "function",
        "function": {
            "name": "ai_agg",
            "description": "Reduce large text columns using natural language instructions. Handles datasets LARGER than LLM context windows (unlike summarize tool). Examples: AI_AGG(reviews, 'Summarize customer feedback'), AI_AGG('Menu: ' || menu_item || '\\nReview: ' || review, 'Find most positive review to highlight on website'), SELECT product_id, AI_AGG(review, 'Identify common complaints') FROM reviews GROUP BY product_id. Use for: aggregating reviews/comments/transcripts across many rows, extracting patterns from large datasets, custom aggregation instructions like 'Describe common complaints', 'Identify all people mentioned with short biographies', 'Find patterns across customer feedback'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction describing how to aggregate the text. Use declarative statements like 'Summarize the reviews', 'Identify common themes', 'Find the most positive review'. Be specific about the intended use case.",
                    },
                    "text": {
                        "type": "string",
                        "description": "Direct text string to aggregate. Use for simple aggregations without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query that returns text data to aggregate. Can include GROUP BY to aggregate by groups. Example: 'SELECT product_id, review FROM reviews'.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column name containing text to aggregate when using 'query' parameter. Required when query is provided.",
                    },
                },
                "required": ["instruction"],
            },
        },
    }
    raw_tools.append(ai_agg_tool)

    # Add ai_summarize_agg tool for general-purpose text summarization
    ai_summarize_agg_tool = {
        "type": "function",
        "function": {
            "name": "ai_summarize_agg",
            "description": "General-purpose summarization of large text columns. Handles datasets LARGER than LLM context windows. Examples: AI_SUMMARIZE_AGG(churn_reason), SELECT restaurant_id, AI_SUMMARIZE_AGG(review) FROM reviews GROUP BY restaurant_id, AI_SUMMARIZE_AGG('Item: ' || item || '\\nReview: ' || text). Automatically generates summaries without needing custom instructions. For specific aggregations with custom prompts like 'identify complaints' or 'find patterns', use ai_agg instead. For single document summarization, use summarize tool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Direct text string to summarize. Use for simple summarizations without a query.",
                    },
                    "query": {
                        "type": "string",
                        "description": "SQL query that returns text data to summarize. Can include GROUP BY to summarize by groups.",
                    },
                    "text_column": {
                        "type": "string",
                        "description": "Column name containing text to summarize when using 'query' parameter. Required when query is provided.",
                    },
                },
                "required": [],
            },
        },
    }
    raw_tools.append(ai_summarize_agg_tool)

    # Add render_chart tool for creating visualizations from query data
    render_chart_tool = {
        "type": "function",
        "function": {
            "name": "render_chart",
            "description": "Render a chart visualization from query results. Use this AFTER execute_query returns data. Pass the ENTIRE rowData array from execute_query result, plus the column names to use for labels and values. The tool extracts the data from the columns automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["pie", "donut", "bar", "line", "scatter", "area"],
                        "description": "Type of chart. Use 'pie' for proportions/percentages, 'bar' for comparisons, 'line' for trends.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title to display.",
                    },
                    "data": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "The rowData array from execute_query results. Pass the entire array of row objects.",
                    },
                    "label_column": {
                        "type": "string",
                        "description": "Name of the column to use for chart labels/categories (e.g., 'SECTOR', 'PRODUCT_NAME').",
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Name of the column to use for chart values/sizes (e.g., 'WEIGHT_PERCENTAGE', 'TOTAL_SALES').",
                    },
                },
                "required": [
                    "chart_type",
                    "title",
                    "data",
                    "label_column",
                    "value_column",
                ],
            },
        },
    }
    raw_tools.append(render_chart_tool)

    normalized_tools: list[Any] = []
    for tool in raw_tools:
        if isinstance(tool, str):
            try:
                parsed = json.loads(tool)
                normalized_tools.append(parsed)
            except json.JSONDecodeError:
                # Fall back to simple wrapper so downstream code can inspect name
                normalized_tools.append({"function": {"name": tool}})
        else:
            normalized_tools.append(tool)

    return normalized_tools


# Backward compatible exports
_last_query_results = ToolState.instance().last_query_results

__all__ = [
    "execute_tool",
    "get_tool_definitions",
    "ToolContext",
    "ToolState",
    "get_last_query_results",
    "set_last_query_result",
    "_last_query_results",
    "TOOL_REGISTRY",
]
