"""Cortex NLP tool handlers (sentiment, summarize, translate)."""

import asyncio
import json
import subprocess
from typing import TYPE_CHECKING

from ..helpers import _find_snow_cli_binary

if TYPE_CHECKING:
    from ..base import ToolContext


async def handle_sentiment(ctx: "ToolContext", args: dict):
    """Handle sentiment tool call."""
    snow_cli = _find_snow_cli_binary()
    if not snow_cli:
        yield "Snowflake CLI not found.", {"error": "Snowflake CLI not found"}
        return

    command = [snow_cli, "cortex", "sentiment", args["text"]]

    try:
        result = await asyncio.to_thread(
            subprocess.run, command, capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        yield result.stdout, data
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        error_msg = f"Error executing sentiment: {e}"
        yield error_msg, {"error": str(e)}


async def handle_summarize(ctx: "ToolContext", args: dict):
    """Handle summarize tool call."""
    snow_cli = _find_snow_cli_binary()
    if not snow_cli:
        yield "Snowflake CLI not found.", {"error": "Snowflake CLI not found"}
        return

    command = [snow_cli, "cortex", "summarize", args["text"]]

    try:
        result = await asyncio.to_thread(
            subprocess.run, command, capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        yield result.stdout, data
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        error_msg = f"Error executing summarize: {e}"
        yield error_msg, {"error": str(e)}


async def handle_translate(ctx: "ToolContext", args: dict):
    """Handle translate tool call."""
    snow_cli = _find_snow_cli_binary()
    if not snow_cli:
        yield "Snowflake CLI not found.", {"error": "Snowflake CLI not found"}
        return

    command = [snow_cli, "cortex", "translate"]
    command.extend(["--text", args["text"], "--to", args["to_lang"]])
    if "from_lang" in args:
        command.extend(["--from", args["from_lang"]])

    try:
        result = await asyncio.to_thread(
            subprocess.run, command, capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        yield result.stdout, data
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        error_msg = f"Error executing translate: {e}"
        yield error_msg, {"error": str(e)}
