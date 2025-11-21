"""Snowflake AI helpers."""

# flake8: noqa: PLR0911, PLR0912
# pylint: disable = R0911, R0912, R0914, R0915, R0917, C0103, C0415

import base64
import io
import json
import os
from typing import Tuple

from openbb_ai.models import (
    DataContent,
    DataFileReferences,
    ImageDataFormat,
    PdfDataFormat,
    RawObjectDataFormat,
    SingleDataContent,
    SingleFileReference,
    SpreadsheetDataFormat,
)

# Store PDF text positions as dictionaries (from pdfplumber extraction)
pdf_text_blocks: dict[str, list[dict]] = {}
# Store which quotes the LLM actually referenced
llm_referenced_quotes: dict[str, list[tuple[str, int]]] = {}


def split_reasoning_and_final(text: str) -> tuple[str | None, str]:
    separators = ["\n---\n", "\r\n---\r\n", "\n---", "---\n"]
    for sep in separators:
        if sep in text:
            reasoning, final = text.split(sep, 1)
            return reasoning.strip(), final.strip()
    return None, text.strip()


def format_thinking_block(content: str) -> str:
    """Wrap reasoning text in a thinking code block."""
    if not content or not content.strip():
        return ""
    return f"```thinking\n{content.strip()}\n```"


def parse_widget_data(
    data: DataContent | DataFileReferences | list | dict | str, conversation_id: str
) -> str:
    """Parse widget data into a readable string format."""
    result_parts = []

    # If it's a string, just return it
    if isinstance(data, str):
        return data

    # Handle dict that might contain widget results
    if isinstance(data, dict):
        # Check if it's a dict of widget UUIDs to results
        for widget_id, widget_result in data.items():
            result_parts.append(f"### Widget: {widget_id}\n")
            if isinstance(widget_result, str):
                result_parts.append(widget_result)
            elif isinstance(widget_result, dict):
                result_parts.append(json.dumps(widget_result, indent=2))
            elif isinstance(widget_result, list):
                for item in widget_result:
                    if hasattr(item, "data_format"):
                        result_parts.append(
                            parse_single_data_item(item, conversation_id)
                        )
                    else:
                        result_parts.append(str(item))
            else:
                result_parts.append(str(widget_result))
            result_parts.append("\n")
        return "\n".join(result_parts)

    # Handle list - this is the main case we're hitting
    if isinstance(data, list):
        for idx, item in enumerate(data):
            result_parts.append(f"### Data Source {idx + 1}\n")

            # Check if it's a Pydantic model with 'items' attribute (DataContent or DataFileReferences)
            if hasattr(item, "items"):
                items_list = getattr(item, "items", [])
                if isinstance(items_list, list):
                    for result_item in items_list:
                        if hasattr(result_item, "data_format"):
                            result_parts.append(
                                parse_single_data_item(result_item, conversation_id)
                            )
                        # Try to get content directly
                        elif hasattr(result_item, "content"):
                            content = getattr(result_item, "content")
                            if isinstance(content, str):
                                result_parts.append(content)
                            elif isinstance(content, (dict, list)):
                                result_parts.append(json.dumps(content, indent=2))
                            else:
                                result_parts.append(str(content))
                        else:
                            result_parts.append(str(result_item))
                else:
                    result_parts.append(str(items_list))

            # Check if item has data_format directly
            elif hasattr(item, "data_format"):
                result_parts.append(parse_single_data_item(item, conversation_id))

            # Check if it's a dict
            elif isinstance(item, dict):
                result_parts.append(json.dumps(item, indent=2))

            # Check if it has content attribute directly
            elif hasattr(item, "content"):
                content = getattr(item, "content")
                if isinstance(content, str):
                    result_parts.append(content)
                elif isinstance(content, (dict, list)):
                    result_parts.append(json.dumps(content, indent=2))
                else:
                    result_parts.append(str(content))

            # Fallback - try to convert to dict if it's a Pydantic model
            elif hasattr(item, "model_dump"):
                try:
                    dumped = item.model_dump()
                    result_parts.append(json.dumps(dumped, indent=2))
                except Exception:
                    result_parts.append(str(item))

            # Last resort
            else:
                result_parts.append(str(item))

            result_parts.append("\n")

        return "\n".join(result_parts)

    # Single item - not a list or dict
    # Check if item has data_format attribute (SingleDataContent or SingleFileReference)
    if hasattr(data, "data_format"):
        return parse_single_data_item(data, conversation_id)

    # Handle items attribute (DataContent or DataFileReferences)
    if hasattr(data, "items"):
        items_list = getattr(data, "items", [])
        if isinstance(items_list, list):
            for result in items_list:
                result_parts.append(parse_single_data_item(result, conversation_id))
            return "\n\n".join(result_parts)

    # Try model_dump for Pydantic models
    if hasattr(data, "model_dump"):
        try:
            dumped = data.model_dump()
            return json.dumps(dumped, indent=2)
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                print(e)  # noqa: T201

    # Fallback
    return str(data)


def extract_pdf_with_positions(pdf_bytes: bytes) -> Tuple[str, list[dict]]:
    """Extract text and character positions from PDF using pdfplumber.

    Returns:
        Tuple of (full_text, text_positions)
        where text_positions contains dicts with text, page, x0, top, x1, bottom
    """
    import pdfplumber

    document_text = ""
    text_positions = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Use extract_words for reliable text extraction with coordinates
            # This avoids the issue of mashed characters when using page.chars directly
            words = page.extract_words(
                keep_blank_chars=False, x_tolerance=3, y_tolerance=3
            )

            # Group words into lines based on 'top' coordinate
            lines = {}
            for word in words:
                # Rounding top to nearest integer helps group words on same line
                top = round(word["top"])
                if top not in lines:
                    lines[top] = []
                lines[top].append(word)

            # Sort lines by vertical position
            sorted_tops = sorted(lines.keys())

            for top in sorted_tops:
                line_words = lines[top]
                # Sort words in line by horizontal position
                line_words.sort(key=lambda w: w["x0"])

                # Reconstruct line text
                line_text = " ".join(w["text"] for w in line_words)

                if len(line_text.strip()) > 5:  # Filter out noise
                    # Calculate bounding box for the line
                    x0 = min(w["x0"] for w in line_words)
                    x1 = max(w["x1"] for w in line_words)
                    bottom = max(w["bottom"] for w in line_words)

                    text_positions.append(
                        {
                            "text": line_text,
                            "page": page_num,
                            "x0": x0,
                            "top": top,
                            "x1": x1,
                            "bottom": bottom,
                        }
                    )

            # Extract full text for context
            page_text = page.extract_text()
            if page_text:
                document_text += page_text + "\n\n"

    return document_text, text_positions


def find_best_match(
    query_text: str,
    pdf_positions: list[dict],
    preferred_page: int | None = None,
) -> dict | None:
    """Find the best matching line in PDF positions for the query text.

    This function finds the PDF line that best matches what the LLM just said.
    """
    import re
    from difflib import SequenceMatcher

    if not query_text or not pdf_positions:
        return None

    query_lower = query_text.lower().strip()

    # Extract ALL meaningful words from the query (what the LLM just wrote)
    # Don't hardcode specific terms - extract what's ACTUALLY in the text!
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
    }

    # Get actual words from what the LLM wrote
    query_words = [
        word
        for word in re.findall(r"\b[a-z]+\b", query_lower)
        if word not in stopwords and len(word) > 2
    ]

    if not query_words:
        return None

    best_match = None
    best_score = 0

    # Check each PDF line for similarity to what the LLM just wrote
    for position in pdf_positions:
        line_lower = position["text"].lower()

        # Calculate how many of the LLM's words appear in this PDF line
        matching_words = sum(1 for word in query_words if word in line_lower)

        # Use sequence matching to find similar phrases
        seq_matcher = SequenceMatcher(None, query_lower, line_lower)
        similarity_ratio = seq_matcher.ratio()

        # Combined score: word matches + sequence similarity
        score = (matching_words * 10) + (similarity_ratio * 100)

        if preferred_page and position.get("page") == preferred_page:
            score *= 1.5  # bias toward hinted page
        if score > best_score:
            best_score = score
            best_match = position

    # Only return a match if it's actually relevant (not just random overlap)
    min_threshold = 30  # Require decent score to avoid random matches

    if best_score < min_threshold:
        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(f"[DEBUG] No good match found for citation. Best score: {best_score}")
            print(f"[DEBUG] Query words: {query_words[:10]}")
        return None

    if os.environ.get("SNOWFLAKE_DEBUG"):
        print(
            f"[DEBUG] Found match with score {best_score}: {best_match['text'][:80]}..."
        )

    return best_match


def parse_single_data_item(result, conversation_id: str) -> str:
    """Parse a single data item (SingleDataContent or SingleFileReference)."""
    data_format = result.data_format

    # Parse PDF content
    if isinstance(data_format, PdfDataFormat):
        if isinstance(result, SingleDataContent):
            try:
                # Decode base64 to bytes
                pdf_bytes = base64.b64decode(result.content)

                # Extract text with positions using pdfplumber
                full_text, text_positions = extract_pdf_with_positions(pdf_bytes)

                # Store text positions for citation generation
                pdf_text_blocks[conversation_id] = text_positions

                return f"[PDF Content]:\n{full_text}"
            except Exception as e:
                # Fallback to original decoding if pdfplumber fails
                try:
                    content = base64.b64decode(result.content).decode(
                        "utf-8", errors="ignore"
                    )
                    return f"[PDF Content]:\n{content}"
                except Exception as fallback_e:
                    return f"[PDF Content - Unable to parse: {e}, fallback failed: {fallback_e}]"
        elif isinstance(result, SingleFileReference):
            return f"[PDF File]: {result.url}"

    # Parse Image data
    elif isinstance(data_format, ImageDataFormat):
        if isinstance(result, SingleDataContent):
            return f"[Image Data - Base64 encoded, {len(result.content)} bytes]"
        if isinstance(result, SingleFileReference):
            return f"[Image File]: {result.url}"

    # Parse Spreadsheet data
    elif isinstance(data_format, SpreadsheetDataFormat):
        if isinstance(result, SingleDataContent):
            return f"[Spreadsheet Data]:\n{result.content}"
        if isinstance(result, SingleFileReference):
            return f"[Spreadsheet File]: {result.url}"

    # Parse RawObject data (JSON/dict)
    elif isinstance(data_format, RawObjectDataFormat):
        if isinstance(result, SingleDataContent):
            try:
                if isinstance(result.content, str):
                    parsed = json.loads(result.content)
                else:
                    parsed = result.content
                formatted = json.dumps(parsed, indent=2)
                return f"[Data Object]:\n{formatted}"
            except Exception:
                return f"[Data Object]:\n{result.content}"
        if isinstance(result, SingleFileReference):
            return f"[Data File]: {result.url}"

    # Fallback for unknown formats
    elif isinstance(result, SingleDataContent):
        return f"[Unknown Format Data]: {str(result.content)[:500]}"
    elif isinstance(result, SingleFileReference):
        return f"[Unknown Format File]: {result.url}"

    return str(result)


def cleanup_text(text: str) -> str:
    """Clean up text from the LLM response."""
    text = text.strip()
    # Remove wrapping quotes
    while (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        text = text[1:-1].strip()

    # Replace escape sequences
    text = text.replace("\\n", "\n")
    text = text.replace("\\t", "")
    text = text.replace('\\"', '"')
    text = text.replace("\\-", "-")

    # Fix bullet points
    text = text.replace("\t-", "-")
    text = text.replace("\t", "")

    # Ensure bullet points are formatted properly
    lines = text.split("\n")
    formatted_lines = []
    for line in lines:
        new_line = line.strip()
        if new_line.startswith("-") and not new_line.startswith("- "):
            new_line = "- " + new_line[1:].strip()
        formatted_lines.append(new_line)

    return "\n".join(formatted_lines)


def cleanup_identifier(identifier: str) -> str:
    """
    Cleans up a Snowflake identifier (like a table name) by removing common
    LLM-added artifacts like quotes, backticks, and leading/trailing whitespace.
    """
    if not identifier:
        return ""

    # Remove leading/trailing whitespace
    cleaned = identifier.strip()

    # Remove surrounding quotes or backticks
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (
        cleaned.startswith('"') and cleaned.endswith('"')
    ):
        cleaned = cleaned[1:-1]

    # A second strip to handle cases like "' table_name '"
    cleaned = cleaned.strip()

    # The LLM sometimes returns markdown-style backticks, remove them
    if cleaned.startswith("`") and cleaned.endswith("`"):
        cleaned = cleaned[1:-1]

    # Remove any remaining backslashes which might escape quotes
    cleaned = cleaned.replace("\\", "")

    return cleaned


def to_sse(sse_event):
    """Convert the SSE event model to a dictionary for EventSourceResponse."""
    return {"event": sse_event.event, "data": json.dumps(sse_event.data.model_dump())}


def find_quote_in_pdf_blocks(quote: str, conversation_id: str) -> dict | None:
    """Find a quote in the PDF text blocks and return its position data."""
    if conversation_id not in pdf_text_blocks:
        return None

    pdf_positions = pdf_text_blocks[conversation_id]
    quote_lower = quote.lower().strip()

    # Try exact match first
    for position in pdf_positions:
        if quote_lower in position["text"].lower():
            return position

    # Try partial match (first 50 chars)
    if len(quote) > 50:
        quote_start = quote_lower[:50]
        for position in pdf_positions:
            if quote_start in position["text"].lower():
                return position

    # Try key words match
    quote_words = quote_lower.split()
    if len(quote_words) > 5:
        key_words = quote_words[:5]  # First 5 words
        for position in pdf_positions:
            text_lower = position["text"].lower()
            if all(word in text_lower for word in key_words):
                return position

    return None


def extract_quotes_from_llm_response(response_text: str) -> list[tuple[str, int]]:
    """Extract citation references from LLM response.

    Returns list of (surrounding_text, citation_number) tuples.
    """
    import re

    citations = []

    # Find all [N] citation references
    pattern = r"([^.!?]*?)\[(\d+)\]"
    matches = re.findall(pattern, response_text)

    for context_text, citation_num in matches:
        # Get the sentence containing this citation
        sentence = context_text.strip()
        if len(sentence) > 200:
            sentence = sentence[-200:]  # Last 200 chars before citation
        citations.append((sentence, int(citation_num)))

    # If no numbered citations found, try old format as fallback
    if not citations:
        # Pattern for "Quote text" (Page X) format
        pattern_old = r'"([^"]+)"\s*\((?:[Pp]age|[Pp]\.?)\s*(\d+)\)'
        matches_old = re.findall(pattern_old, response_text)
        for idx, (quote, page) in enumerate(matches_old, 1):
            citations.append((quote, idx))  # Use index as citation number

    return citations


def create_citations_from_widget_data(
    widget_data: list,
    widgets_primary: list,
    input_arguments: dict,
    specific_citation_num: int = None,
):
    """Create citations with PDF highlighting from widget data."""
    from openbb_ai import cite
    from openbb_ai.models import CitationHighlightBoundingBox

    citations_list = []
    conversation_id = input_arguments.get("conversation_id", "default")

    if not widget_data or not widgets_primary:
        return citations_list

    # Get the widget that contains the PDF
    widget = widgets_primary[0] if widgets_primary else None
    if not widget:
        return citations_list

    # Get stored PDF text positions
    if conversation_id not in pdf_text_blocks:
        return citations_list

    pdf_positions = pdf_text_blocks[conversation_id]
    if not pdf_positions:
        return citations_list

    # For specific citation number, find what the LLM is ACTUALLY citing
    if specific_citation_num is not None:
        selected_position = None

        # Check what the LLM was referencing when it added [N]
        if conversation_id in llm_referenced_quotes:
            # Find the context for THIS citation number
            for context_text, cite_num in llm_referenced_quotes[conversation_id]:
                if cite_num == specific_citation_num:
                    # The LLM was talking about context_text when it added [N]
                    # Find the BEST matching position in the PDF
                    import re

                    # Get the last sentence(s)
                    sentences = re.split(r"(?<=[.!?])\s+", context_text)
                    sentences = [s for s in sentences if s.strip()]

                    if sentences:
                        query = sentences[-1]
                        # If last sentence is very short, take previous too
                        if len(query) < 50 and len(sentences) > 1:
                            query = sentences[-2] + " " + query

                        selected_position = find_best_match(query, pdf_positions)

                        if selected_position and os.environ.get("SNOWFLAKE_DEBUG"):
                            print(
                                f"[DEBUG] Matched citation [{specific_citation_num}] to page {selected_position['page']}"
                            )
                    break

        # Create the citation object
        citation_obj = cite(
            widget=widget,
            input_arguments=input_arguments,
            extra_details={
                "Page": selected_position["page"] if selected_position else 1,
                "Reference": (
                    selected_position["text"][:100] + "..."
                    if selected_position and len(selected_position["text"]) > 100
                    else (
                        selected_position["text"]
                        if selected_position
                        else "Document Reference"
                    )
                ),
            },
        )

        # Only add bounding box if we actually found a match
        if selected_position:
            citation_obj.quote_bounding_boxes = [
                [
                    CitationHighlightBoundingBox(
                        text=selected_position["text"],
                        page=selected_position["page"],
                        x0=selected_position["x0"],
                        top=selected_position["top"],
                        x1=selected_position["x1"],
                        bottom=selected_position["bottom"],
                    )
                ]
            ]

        citations_list.append(citation_obj)

        if os.environ.get("SNOWFLAKE_DEBUG"):
            print(
                f"[DEBUG] Created citation [{specific_citation_num}] -> "
                f"widget: {widget.uuid if hasattr(widget, 'uuid') else 'unknown'}, "
                f"match found: {selected_position is not None}"
            )

    return citations_list
