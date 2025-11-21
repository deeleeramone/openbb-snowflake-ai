"""Snowflake AI - Python bindings for Snowflake AI_COMPLETE."""

# Import the Rust extension
try:
    from ._snowflake_ai import SnowflakeAI
except ImportError as e:
    raise ImportError(
        "The snowflake_ai extension module is not installed. "
        "Please build it with:\n"
        "  cd /Users/darrenlee/github/OpenBB/desktop/snowflake\n"
        "  maturin develop --release\n"
        f"Original error: {e}"
    )

from openbb_core.env import Env

_ = Env()

# Import server components
try:
    from .server import create_app, run_server

    __all__ = [
        "SnowflakeAI",
        "create_app",
        "run_server",
    ]
except ImportError:
    __all__ = [
        "SnowflakeAI",
    ]

__version__ = "0.1.0"
