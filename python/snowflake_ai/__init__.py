"""Snowflake AI - Python bindings for Snowflake AI_COMPLETE."""

from openbb_core.env import Env

_ = Env()

# Import the Rust extension
try:
    from ._snowflake_ai import SnowflakeAI, SnowflakeAgent
except ImportError as e:
    raise ImportError(
        "The snowflake_ai extension module is not installed. "
        "Please build it from source with:\n"
        "maturin develop --release\n"
        f"Original error: {e}"
    ) from None

# Import server components
try:
    from . import (
        conversation_manager,
        document_processor,
        helpers,
        logger,
        models,
        server,
        slash_commands,
        streaming_handler,
        system_prompt,
        tool_executor,
        widget_handler,
        widgets,
    )

    __all__ = [
        "SnowflakeAI",
        "SnowflakeAgent",
        "conversation_manager",
        "document_processor",
        "helpers",
        "logger",
        "models",
        "server",
        "slash_commands",
        "streaming_handler",
        "system_prompt",
        "tool_executor",
        "widget_handler",
        "widgets",
    ]
except ImportError:
    __all__ = [
        "SnowflakeAI",
        "SnowflakeAgent",
    ]

__version__ = "0.1.0"
