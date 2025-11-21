"""Widget data caching for OpenBB widget protocol."""

import hashlib
import json
import os
from typing import Any, Dict, Optional
from datetime import datetime, timedelta


class WidgetDataCache:
    """Cache manager for widget data following OpenBB protocol."""

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl_seconds = 300  # 5 minutes cache TTL

    def _generate_cache_key(self, widget_id: str, conv_id: str) -> str:
        """Generate cache key based on widget ID and conversation."""
        key_string = f"{conv_id}|{widget_id}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self, widget_data: dict[str, Any], conv_id: str
    ) -> Optional[dict[str, Any]]:
        """Retrieve cached widget data if available."""
        # This method needs widget_data to be a dict, but _generate_cache_key expects widget_id string
        # Extract widget_id from widget_data
        widget_id = widget_data.get("widgetId", "")
        if not widget_id:
            return None

        cache_key = self._generate_cache_key(widget_id, conv_id)

        if cache_key in self._cache:
            cached_entry = self._cache[cache_key]
            if datetime.now() < cached_entry["expires_at"]:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Widget cache hit for {widget_id}")
                return cached_entry["data"]
            else:
                del self._cache[cache_key]

        return None

    def set(
        self, widget_data: dict[str, Any], conv_id: str, parsed_data: dict[str, Any]
    ):
        """Store widget data in cache."""
        # Extract widget_id from widget_data
        widget_id = widget_data.get("widgetId", "")
        if not widget_id:
            return

        cache_key = self._generate_cache_key(widget_id, conv_id)

        self._cache[cache_key] = {
            "data": parsed_data,
            "expires_at": datetime.now() + timedelta(seconds=self._ttl_seconds),
            "widget_info": {
                "widgetId": widget_id,
                "type": widget_data.get("type"),
                "title": widget_data.get("title"),
            },
        }

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Cached widget data for {widget_id}")

    def clear_conversation(self, conv_id: str):
        """Clear all cached data for a conversation."""
        keys_to_remove = []
        for key in self._cache:
            # Check if this cache entry belongs to the conversation
            for entry_key in self._cache.keys():
                if conv_id in entry_key:
                    keys_to_remove.append(entry_key)

        for key in keys_to_remove:
            del self._cache[key]

    def get_by_widget_id(self, widget_id: str, conv_id: str) -> Optional[Any]:
        """Retrieve cached widget data by widget ID."""
        cache_key = self._generate_cache_key(widget_id, conv_id)

        if cache_key in self._cache:
            cached_entry = self._cache[cache_key]
            if datetime.now() < cached_entry["expires_at"]:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    print(f"[DEBUG] Widget cache hit for widget {widget_id}")
                return cached_entry["data"]
            else:
                del self._cache[cache_key]

        return None

    def set_by_widget_id(self, widget_id: str, conv_id: str, data: Any):
        """Store widget data in cache by widget ID."""
        cache_key = self._generate_cache_key(widget_id, conv_id)

        self._cache[cache_key] = {
            "data": data,
            "expires_at": datetime.now() + timedelta(seconds=self._ttl_seconds),
            "widget_id": widget_id,
        }

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] Cached widget data for widget {widget_id}")


# Global cache instance
widget_cache = WidgetDataCache()
