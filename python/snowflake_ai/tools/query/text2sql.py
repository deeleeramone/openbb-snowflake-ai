"""Text2SQL tool handler."""

import asyncio
import json
import os
from typing import TYPE_CHECKING

from openbb_ai import reasoning_step

from ..helpers import _get_database_context_for_llm
from ...helpers import to_sse
from ...logger import get_logger

if TYPE_CHECKING:
    from ..base import ToolContext

logger = get_logger(__name__)


async def handle_text2sql(ctx: "ToolContext", args: dict):
    """Handle text2sql tool call."""
    prompt = args.get("prompt") or args.get("question")
    if not prompt:
        yield "Error: 'prompt' argument is required for text2sql.", {
            "error": "Missing prompt"
        }
        return

    conv_id = args.get("_conversation_id", ctx.conv_id)

    # Set conversation context (database/schema) if stored in settings
    # This allows text2sql to use the correct schema from /use_database, /use_schema commands
    try:
        settings_json = await asyncio.to_thread(
            ctx.client.get_or_create_conversation, conv_id, json.dumps({})
        )
        if settings_json:
            settings = json.loads(settings_json)
            db = settings.get("database")
            sch = settings.get("schema")
            if db or sch:
                await asyncio.to_thread(ctx.client.use_conversation_context, db, sch)
    except Exception:
        pass  # Ignore errors, use default database/schema

    # Check if semantic model needs to be generated
    # This is indicated by not having a cached model
    has_cached_model = await asyncio.to_thread(ctx.client.has_semantic_model_cache)

    if not has_cached_model:
        yield to_sse(
            reasoning_step(
                "Preparing semantic model and generating SQL (first run may take 30-60 seconds)...",
                event_type="INFO",
            )
        )
    else:
        yield to_sse(
            reasoning_step(
                "Generating SQL using cached semantic model...",
                event_type="INFO",
            )
        )

    sql_text = ""
    explanation = ""
    request_id = None
    use_llm_fallback = False

    try:
        response = await asyncio.to_thread(ctx.client.text2sql, prompt)
        parsed = json.loads(response)
        sql_text = (parsed.get("sql") or "").strip()
        explanation = (parsed.get("explanation") or "").strip()
        request_id = parsed.get("request_id")
    except Exception as exc:  # pragma: no cover - defensive
        error_str = str(exc)
        if "ALL_VIEWS_EXHAUSTED" in error_str:
            # All semantic views failed - trigger LLM fallback
            yield to_sse(
                reasoning_step(
                    "Semantic views exhausted. Falling back to LLM-generated SQL...",
                    event_type="WARNING",
                )
            )
            use_llm_fallback = True
        else:
            error_msg = f"text2sql generation failed: {exc}"
            logger.error(error_msg)
            yield error_msg, {"error": str(exc)}
            return

    # LLM Fallback: Generate SQL using conversation context and schema
    if use_llm_fallback or not sql_text:
        yield to_sse(
            reasoning_step(
                "Using LLM to generate SQL from database schema...",
                event_type="INFO",
            )
        )

        try:
            # Get database context for the LLM
            db_context = await _get_database_context_for_llm(ctx.client, prompt)

            # Build LLM prompt for SQL generation
            llm_prompt = f"""You are a Snowflake SQL expert. Generate a valid Snowflake SQL query to answer the user's question.

Database Context:
{db_context}

User Question: {prompt}

Rules:
1. Return ONLY a valid Snowflake SQL query, nothing else
2. Use fully qualified table names (DATABASE.SCHEMA.TABLE)
3. Use proper Snowflake SQL syntax
4. Do NOT include markdown code blocks or backticks
5. If you cannot generate a valid query, explain why briefly

SQL Query:"""

            # Use CORTEX.COMPLETE to generate SQL
            llm_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('llama3.1-70b', $${llm_prompt}$$) as RESULT"
            result = await asyncio.to_thread(ctx.client.execute_query, llm_query)
            result_json = json.loads(result)

            if result_json.get("rowData"):
                llm_result = result_json["rowData"][0].get("RESULT", "")
                # Clean up the result - remove any markdown formatting
                llm_result = llm_result.strip()
                if llm_result.startswith("```"):
                    # Remove markdown code blocks
                    lines = llm_result.split("\n")
                    llm_result = "\n".join(
                        line for line in lines if not line.startswith("```")
                    ).strip()

                # Check if it looks like valid SQL
                sql_keywords = [
                    "SELECT",
                    "INSERT",
                    "UPDATE",
                    "DELETE",
                    "WITH",
                    "CREATE",
                    "ALTER",
                    "DROP",
                ]
                if any(llm_result.upper().startswith(kw) for kw in sql_keywords):
                    sql_text = llm_result
                    explanation = "SQL generated by LLM fallback (semantic views could not process this query)"
                else:
                    # LLM returned an explanation instead of SQL
                    explanation = llm_result
                    sql_text = ""
        except Exception as llm_exc:
            logger.error("LLM fallback failed: %s", llm_exc)
            explanation = f"Failed to generate SQL: {llm_exc}"
            sql_text = ""

    output_sections: list[str] = []
    if explanation:
        output_sections.append(explanation)
    if sql_text:
        output_sections.append(f"```sql\n{sql_text}\n```")
    else:
        output_sections.append("No SQL was generated.")
    if request_id:
        output_sections.append(f"(request_id: {request_id})")

    yield "\n\n".join(output_sections), {
        "sql": sql_text,
        "explanation": explanation,
        "request_id": request_id,
    }
