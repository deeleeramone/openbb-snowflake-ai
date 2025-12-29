"""Semantic views tool handler."""

import asyncio
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_list_semantic_views(ctx: "ToolContext", args: dict):
    """Handle list_semantic_views tool call."""
    try:
        result_json = await asyncio.to_thread(ctx.client.list_semantic_views)
        views = json.loads(result_json)
        if not views:
            yield "No semantic views found in the current database.", {"views": []}
            return

        # Format with FULL details from DESCRIBE SEMANTIC VIEW
        output_parts = ["**Semantic Views in Database:**\n"]

        for v in views:
            name = v.get("name", "")
            fqn = v.get("fqn", "")
            comment = v.get("comment") or "(no comment)"
            owner = v.get("owner") or ""

            output_parts.append(f"### {name}")
            output_parts.append(f"- **FQN:** `{fqn}`")
            output_parts.append(f"- **Comment:** {comment}")
            output_parts.append(f"- **Owner:** {owner}")

            # Tables
            tables = v.get("tables", [])
            if tables:
                output_parts.append(f"- **Tables ({len(tables)}):**")
                for t in tables:
                    base = t.get("base_table", "")
                    base_db = t.get("base_database", "")
                    base_sch = t.get("base_schema", "")
                    pk = t.get("primary_key", "")
                    output_parts.append(
                        f"  - `{t.get('name', '')}` → {base_db}.{base_sch}.{base}"
                        + (f" (PK: {pk})" if pk else "")
                    )

            # Dimensions
            dims = v.get("dimensions", [])
            if dims:
                output_parts.append(f"- **Dimensions ({len(dims)}):**")
                for d in dims:
                    expr = d.get("expression", "")
                    dtype = d.get("data_type", "")
                    output_parts.append(f"  - `{d.get('name', '')}`: {expr} ({dtype})")

            # Facts
            facts = v.get("facts", [])
            if facts:
                output_parts.append(f"- **Facts ({len(facts)}):**")
                for f in facts:
                    expr = f.get("expression", "")
                    dtype = f.get("data_type", "")
                    output_parts.append(f"  - `{f.get('name', '')}`: {expr} ({dtype})")

            # Metrics
            metrics = v.get("metrics", [])
            if metrics:
                output_parts.append(f"- **Metrics ({len(metrics)}):**")
                for m in metrics:
                    expr = m.get("expression", "")
                    dtype = m.get("data_type", "")
                    output_parts.append(f"  - `{m.get('name', '')}`: {expr} ({dtype})")

            output_parts.append("")  # blank line between views

        yield "\n".join(output_parts), {"views": views, "_raw": True}

    except Exception as e:
        error_msg = f"Error listing semantic views: {str(e)}"
        yield error_msg, {"error": str(e)}
