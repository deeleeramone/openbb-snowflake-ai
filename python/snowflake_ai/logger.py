"""Logging configuration for Snowflake AI."""

import logging
import os
import sys

# Create logger
logger = logging.getLogger("snowflake_ai")

# Configure logger based on environment
debug_mode = os.environ.get("SNOWFLAKE_DEBUG", "").lower() in ("1", "true", "yes")

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Create console handler with formatting
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

# Create formatter - simplified for production
if debug_mode:
    formatter = logging.Formatter(
        "[%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s"
    )
else:
    formatter = logging.Formatter("[%(levelname)s] %(message)s")

handler.setFormatter(formatter)

# Add handler to logger (avoid duplicates)
if not logger.handlers:
    logger.addHandler(handler)

# Prevent propagation to root logger
logger.propagate = False


def get_logger(name: str = "snowflake_ai") -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to snowflake_ai)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
