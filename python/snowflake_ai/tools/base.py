"""Base types and shared state for tool execution."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator

if TYPE_CHECKING:
    from .._snowflake_ai import SnowflakeAI


@dataclass
class ToolContext:
    """Context passed to each tool execution."""

    client: "SnowflakeAI"
    conv_id: str = "default"


class ToolState:
    """Singleton for shared state across tool invocations."""

    _instance: "ToolState | None" = None
    _last_query_results: dict[str, list]

    def __new__(cls) -> "ToolState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._last_query_results = {}
        return cls._instance

    @property
    def last_query_results(self) -> dict[str, list]:
        """Get the last query results cache."""
        return self._last_query_results

    @classmethod
    def instance(cls) -> "ToolState":
        """Get the singleton instance."""
        return cls()


# Module-level access for backward compatibility
_tool_state = ToolState.instance()


def get_last_query_results() -> dict[str, list]:
    """Get the last query results cache (backward compatible)."""
    return _tool_state.last_query_results


def set_last_query_result(conv_id: str, data: list) -> None:
    """Store query results for a conversation."""
    # pylint: disable=import-outside-toplevel
    import copy
    from ..logger import get_logger

    logger = get_logger(__name__)

    # Deep copy to prevent data mutation issues
    copied_data = copy.deepcopy(data)
    _tool_state.last_query_results[conv_id] = copied_data

    if copied_data:
        first_row_preview = str(copied_data[0])[:200] if copied_data else "EMPTY"
        logger.debug(
            "[SINGLETON STORE] conv_id=%s, rows=%d, first=%s",
            conv_id,
            len(copied_data),
            first_row_preview,
        )
    else:
        logger.debug("[SINGLETON STORE] conv_id=%s, rows=0 (empty)", conv_id)


# Type alias for tool result generators
ToolResult = AsyncGenerator[tuple[str, dict[str, Any]] | str, None]
