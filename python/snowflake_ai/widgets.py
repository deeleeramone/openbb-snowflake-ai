"""Widget endpoints for Snowflake AI OpenBB Workspace integration."""

import json
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from openbb_platform_api.response_models import OmniWidgetResponseModel

from ._snowflake_ai import SnowflakeAI

router = APIRouter(prefix="/widgets", tags=["widgets"])


class SnowflakeClient:
    """Lazily instantiate and reuse a SnowflakeAI connection for widget calls."""

    def __init__(self):
        self._client: SnowflakeAI | None = None

    def _build_client(self) -> SnowflakeAI:
        required_env = {
            "SNOWFLAKE_USER": os.environ.get("SNOWFLAKE_USER"),
            "SNOWFLAKE_PASSWORD": os.environ.get("SNOWFLAKE_PASSWORD"),
            "SNOWFLAKE_ACCOUNT": os.environ.get("SNOWFLAKE_ACCOUNT"),
            "SNOWFLAKE_ROLE": os.environ.get("SNOWFLAKE_ROLE"),
        }
        missing = [key for key, value in required_env.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required Snowflake credentials: " + ", ".join(sorted(missing))
            )

        return SnowflakeAI(
            user=required_env["SNOWFLAKE_USER"],
            password=required_env["SNOWFLAKE_PASSWORD"],
            account=required_env["SNOWFLAKE_ACCOUNT"],
            role=required_env["SNOWFLAKE_ROLE"],
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE") or "",
            database=os.environ.get("SNOWFLAKE_DATABASE") or "",
            schema=os.environ.get("SNOWFLAKE_SCHEMA") or "",
        )

    def __call__(self) -> SnowflakeAI:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def close(self) -> None:
        """Close the Snowflake client connection."""
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None


snowflake_client = SnowflakeClient()


def close_widget_client() -> None:
    """Close the Snowflake client on shutdown."""
    snowflake_client.close()


router.add_event_handler("shutdown", close_widget_client)


@router.post("/execute", response_model=OmniWidgetResponseModel)
def execute_widget_query(
    payload: dict,
    client: Annotated[SnowflakeAI, Depends(snowflake_client)],
):
    """Execute a Snowflake SQL query."""
    try:
        raw_result = client.execute_query(payload.get("prompt", ""))
    except Exception as exc:  # pragma: no cover - surfaced via HTTPException
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    try:
        parsed_result = json.loads(raw_result)
    except json.JSONDecodeError:
        parsed_result = raw_result

    row_data = parsed_result.get("rowData", [])

    if not row_data:
        raise HTTPException(
            status_code=500, detail="No row data returned from Snowflake query."
        )

    return {"content": row_data}
