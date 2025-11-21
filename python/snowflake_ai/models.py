"""Data models for Snowflake AI API."""

from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A chat message."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="The role of the message sender"
    )
    content: str = Field(..., description="The content of the message")


class ChatRequest(BaseModel):
    """Request for chat completion."""

    messages: list[ChatMessage] = Field(..., description="The conversation history")
    model: str = Field(default="snowflake-arctic", description="The AI model to use")
    temperature: float | None = Field(
        default=0.7, ge=0.0, le=1.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=4096, gt=0, description="Maximum tokens to generate"
    )
    stream: bool = Field(default=True, description="Whether to stream the response")
    context: str | None = Field(
        default=None, description="Additional context to prepend to the conversation"
    )


class ChatResponse(BaseModel):
    """Response from chat completion."""

    id: str = Field(..., description="Unique identifier for the response")
    model: str = Field(..., description="The model used")
    choices: list[dict] = Field(..., description="The generated completions")
    usage: dict | None = Field(None, description="Token usage information")


class SchemaItem(BaseModel):
    """Pydantic model for a schema item."""

    TABLE_CATALOG: str
    TABLE_SCHEMA: str
    TABLE_NAME: str
    COLUMN_NAME: str
    ORDINAL_POSITION: int
    COLUMN_DEFAULT: str | None = None
    IS_NULLABLE: str
    DATA_TYPE: str
