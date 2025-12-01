"""Document processing singleton for Snowflake AI."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import threading
import traceback
from collections.abc import Mapping
from difflib import SequenceMatcher
from typing import Any, Tuple

import pdfplumber
from pdfminer.pdfpage import PDFPage
from pdfminer.pdftypes import PDFObjRef, resolve1
from pdfminer.psparser import PSLiteral

try:
    from pdfminer.pdfdocument import PDFNoOutlines
except ImportError:
    # Fallback for older pdfminer versions
    PDFNoOutlines = Exception  # type: ignore[misc, assignment]

from .logger import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """Centralized handler for PDF extraction, caching, and Snowflake storage."""

    _instance: "DocumentProcessor | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._pdf_text_blocks: dict[str, list[dict]] = {}
        self._snowflake_document_pages: dict[str, list[dict]] = {}
        self._llm_referenced_quotes: dict[str, list[tuple[str, int]]] = {}
        self._document_sources: dict[str, dict] = {}
        self._pdf_metadata: dict[str, dict] = (
            {}
        )  # Cache PDF metadata/outline per conversation
        self._locks: dict[str, asyncio.Lock] = {}

    @classmethod
    def instance(cls) -> "DocumentProcessor":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Conversation cache helpers
    # ------------------------------------------------------------------
    def get_pdf_blocks(self, conversation_id: str) -> list[dict] | None:
        """Get cached PDF text blocks for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier

        Returns
        -------
        list[dict] | None
            List of text block dictionaries or None if not cached
        """
        return self._pdf_text_blocks.get(conversation_id)

    def set_pdf_blocks(self, conversation_id: str, positions: list[dict]) -> None:
        """Cache PDF text blocks for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        positions : list[dict]
            List of text block dictionaries with position data
        """
        self._pdf_text_blocks[conversation_id] = positions

    @property
    def pdf_text_store(self) -> dict[str, list[dict]]:
        """Get the complete PDF text blocks store.

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping conversation IDs to PDF text blocks
        """
        return self._pdf_text_blocks

    def get_document_pages(self, conversation_id: str) -> list[dict] | None:
        """Get cached Snowflake document pages for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier

        Returns
        -------
        list[dict] | None
            List of page dictionaries or None if not cached
        """
        return self._snowflake_document_pages.get(conversation_id)

    def set_document_pages(self, conversation_id: str, pages: list[dict]) -> None:
        """Cache Snowflake document pages for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        pages : list[dict]
            List of page dictionaries with content
        """
        self._snowflake_document_pages[conversation_id] = pages

    @property
    def snowflake_page_store(self) -> dict[str, list[dict]]:
        """Get the complete Snowflake document pages store.

        Returns
        -------
        dict[str, list[dict]]
            Dictionary mapping conversation IDs to document pages
        """
        return self._snowflake_document_pages

    def record_llm_quote(
        self, conversation_id: str, quote: str, citation_num: int
    ) -> None:
        """Record a quote referenced by the LLM with its citation number.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        quote : str
            The quoted text from the LLM response
        citation_num : int
            The citation number assigned to this quote
        """
        self._llm_referenced_quotes.setdefault(conversation_id, []).append(
            (quote, citation_num)
        )

    def get_llm_quotes(self, conversation_id: str) -> list[tuple[str, int]]:
        """Get all LLM-referenced quotes for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier

        Returns
        -------
        list[tuple[str, int]]
            List of tuples (quote_text, citation_number)
        """
        return list(self._llm_referenced_quotes.get(conversation_id, []))

    def clear_llm_quotes(self, conversation_id: str) -> None:
        """Clear all LLM-referenced quotes for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        """
        self._llm_referenced_quotes.pop(conversation_id, None)

    def get_pdf_metadata(self, conversation_id: str) -> dict | None:
        """Get cached PDF metadata for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier

        Returns
        -------
        dict | None
            PDF metadata dict or None if not cached
        """
        return self._pdf_metadata.get(conversation_id)

    def search_pdf_metadata(
        self, conversation_id: str, query: str, file_name: str | None = None
    ) -> list[dict]:
        """Search PDF metadata (outline, page summaries) for matching pages.

        This is a fast search method that uses pre-extracted document structure.
        Note: Tables are searched separately via DOCUMENT_PARSE_RESULTS query.

        Search order:
        1. Outline/TOC entries
        2. Page summaries/headers

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        query : str
            Search query (keywords to find)
        file_name : str | None
            Optional filename for context

        Returns
        -------
        list[dict]
            List of matching results with page numbers and snippets
        """
        metadata = self.get_pdf_metadata(conversation_id)
        if not metadata:
            return []

        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # STEP 1: Search outline/TOC
        outline = metadata.get("outline", [])
        for item in outline:
            title = item.get("title", "")
            title_lower = title.lower()

            # Check for any query word match
            if any(word in title_lower for word in query_words):
                results.append(
                    {
                        "file_name": file_name
                        or metadata.get("info", {}).get("title", "document"),
                        "page_number": item.get("page", 1),
                        "chunk_text": f"[Section/TOC Entry] {title}",
                        "similarity_score": 0.9,
                        "match_type": "outline",
                        "source": "pdf_metadata",
                    }
                )

        # STEP 2: Search page summaries
        page_summaries = metadata.get("page_summaries", [])
        for summary in page_summaries:
            first_line = summary.get("first_line", "")
            first_line_lower = first_line.lower()

            if any(word in first_line_lower for word in query_words):
                results.append(
                    {
                        "file_name": file_name
                        or metadata.get("info", {}).get("title", "document"),
                        "page_number": summary.get("page", 1),
                        "chunk_text": f"[Page Header] {first_line}",
                        "similarity_score": 0.8,
                        "match_type": "page_summary",
                        "source": "pdf_metadata",
                    }
                )

        # Remove duplicate pages (keep highest score)
        seen_pages = {}
        for result in results:
            page = result["page_number"]
            if (
                page not in seen_pages
                or result["similarity_score"] > seen_pages[page]["similarity_score"]
            ):
                seen_pages[page] = result

        return sorted(seen_pages.values(), key=lambda x: -x["similarity_score"])

    def set_pdf_metadata(self, conversation_id: str, metadata: dict) -> None:
        """Cache PDF metadata for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        metadata : dict
            PDF metadata dictionary
        """
        self._pdf_metadata[conversation_id] = metadata

    def format_document_structure_for_llm(self, conversation_id: str) -> str:
        """Format PDF metadata as a string for the LLM system prompt.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier

        Returns
        -------
        str
            Formatted document structure string or empty string if no metadata
        """
        metadata = self.get_pdf_metadata(conversation_id)
        if not metadata:
            return ""

        lines = []
        lines.append(f"DOCUMENT STRUCTURE ({metadata.get('page_count', '?')} pages):")

        # Add PDF info if available
        info = metadata.get("info", {})
        if info.get("title"):
            lines.append(f"  Title: {info['title']}")

        # Add outline/TOC if available
        outline = metadata.get("outline", [])
        if outline:
            lines.append("  TABLE OF CONTENTS:")
            for item in outline[:15]:  # Limit to first 15 items
                indent = "    " * item.get("level", 1)
                lines.append(f"{indent}- {item['title']} (page {item['page']})")
            if len(outline) > 15:
                lines.append(f"    ... and {len(outline) - 15} more sections")
        else:
            # No TOC - show page summaries instead
            lines.append("  PAGE OVERVIEW:")
            summaries = metadata.get("page_summaries", [])
            for summary in summaries[:10]:  # First 10 pages
                first_line = summary.get("first_line", "")
                if first_line:
                    lines.append(f"    Page {summary['page']}: {first_line}")
            if len(summaries) > 10:
                lines.append(f"    ... and {len(summaries) - 10} more pages")

        return "\n".join(lines)

    @property
    def llm_quote_store(self) -> dict[str, list[tuple[str, int]]]:
        """Get the complete LLM quotes store.

        Returns
        -------
        dict[str, list[tuple[str, int]]]
            Dictionary mapping conversation IDs to lists of (quote, citation_num) tuples
        """
        return self._llm_referenced_quotes

    def _get_lock(self, key: str) -> asyncio.Lock:
        """Get or create an asyncio lock for a given key.

        Parameters
        ----------
        key : str
            The lock identifier

        Returns
        -------
        asyncio.Lock
            An asyncio.Lock instance
        """
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    async def _run_in_thread(self, func, *args, **kwargs):
        """Run a synchronous function in a thread pool.

        Parameters
        ----------
        func : callable
            The function to execute
        *args
            Positional arguments for the function
        **kwargs
            Keyword arguments for the function

        Returns
        -------
        Any
            The function's return value
        """
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _get_user_schema(self, client) -> str:
        """Get the user-specific schema name in Snowflake.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        str
            Schema name in format USER_{sanitized_username}
        """
        snowflake_user = await self._run_in_thread(client.get_current_user)
        sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
        return f"USER_{sanitized_user}".upper()

    @staticmethod
    def _get_row_value(
        row: Mapping[str, Any] | None,
        *keys: str,
        default: Any | None = None,
    ) -> Any | None:
        """Fetch a value from a Snowflake row mapping, matching keys case-insensitively."""
        if not isinstance(row, Mapping):
            return default

        lowered = {str(k).lower(): v for k, v in row.items()}

        for key in keys:
            if not key:
                continue
            lookup_key = key.lower()
            if lookup_key in lowered:
                value = lowered[lookup_key]
                if value not in (None, ""):
                    return value
                if value is not None:
                    return value

        return default

    # ------------------------------------------------------------------
    # Widget type detection
    # ------------------------------------------------------------------

    @staticmethod
    def extract_document_stage_path(widget) -> str | None:
        """Extract the Snowflake stage path from a document widget.

        Parameters
        ----------
        widget : Widget
            The widget object to extract from

        Returns
        -------
        str | None
            Stage path string (e.g., @DB.SCHEMA.STAGE/file.pdf) or None
        """
        if not widget or not hasattr(widget, "params"):
            return None

        for param in widget.params:
            if param.name == "document":
                current_value = getattr(param, "current_value", None)
                if current_value:
                    if isinstance(current_value, str) and current_value.startswith("@"):
                        return current_value
                    elif isinstance(current_value, list) and current_value:
                        first_val = current_value[0]
                        if isinstance(first_val, str) and first_val.startswith("@"):
                            return first_val

        return None

    # ------------------------------------------------------------------
    # Filename helpers
    # ------------------------------------------------------------------
    @staticmethod
    def extract_filename_from_stage_path(stage_path: str) -> str | None:
        """Extract filename from a Snowflake stage path.

        Parameters
        ----------
        stage_path : str
            The stage path (e.g., @DB.SCHEMA.STAGE/file.pdf)

        Returns
        -------
        str | None
            Filename string or None if extraction fails
        """
        if not stage_path:
            return None
        path = stage_path.lstrip("@")
        if "/" in path:
            return path.split("/")[-1]
        return None

    @staticmethod
    def extract_filename_from_widget(widget) -> str | None:
        """Extract filename from a widget's parameters.

        Parameters
        ----------
        widget : Widget
            The widget object

        Returns
        -------
        str | None
            Filename string or None if not found
        """
        if not hasattr(widget, "params") or not widget.params:
            return None
        for param in widget.params:
            current_value = getattr(param, "current_value", None)
            if current_value and isinstance(current_value, str):
                if current_value.startswith("@") and "/" in current_value:
                    filename = DocumentProcessor.extract_filename_from_stage_path(
                        current_value
                    )
                    if filename and (filename.endswith(".pdf") or "." in filename):
                        return filename
        return None

    @staticmethod
    def extract_pdf_with_positions(
        pdf_bytes: bytes, extract_metadata: bool = False
    ) -> Tuple[str, list[dict], dict | None]:
        """Extract text, bounding box positions, and optionally metadata from a PDF.

        Single-pass extraction that opens the PDF once and extracts everything needed.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF file bytes
        extract_metadata : bool, optional
            Whether to also extract outline/TOC metadata, by default False

        Returns
        -------
        tuple[str, list[dict], dict | None]
            Tuple of (full_text, list of position dictionaries, metadata dict or None).
            Position dict contains: page, text, x0, top, x1, bottom
            Metadata dict contains: page_count, outline, page_summaries, info
        """
        document_text = ""
        text_positions: list[dict] = []
        metadata: dict | None = None
        outline_extracted = False

        if extract_metadata:
            metadata = {
                "page_count": 0,
                "outline": [],
                "page_summaries": [],
                "info": {},
            }

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if extract_metadata and metadata is not None:
                metadata["page_count"] = len(pdf.pages)

                # Extract PDF info/metadata
                if hasattr(pdf, "metadata") and pdf.metadata:
                    info = pdf.metadata
                    metadata["info"] = {
                        "title": info.get("Title", "") or "",
                        "author": info.get("Author", "") or "",
                        "subject": info.get("Subject", "") or "",
                        "keywords": info.get("Keywords", "") or "",
                        "creator": info.get("Creator", "") or "",
                    }

                # Build page_id -> page_number mapping for outline resolution
                page_id_to_num: dict[int, int] = {}
                try:
                    for page_num, pdfminer_page in enumerate(
                        PDFPage.create_pages(pdf.doc), start=1
                    ):
                        page_id_to_num[id(pdfminer_page)] = page_num
                        if hasattr(pdfminer_page, "pageid") and isinstance(
                            pdfminer_page.pageid, int
                        ):
                            page_id_to_num[pdfminer_page.pageid] = page_num
                except Exception:
                    pass

                # Helper to resolve destination to page number
                def resolve_dest_to_page(dest) -> int | None:
                    if dest is None:
                        return None
                    try:
                        if isinstance(dest, PDFObjRef):
                            dest = resolve1(dest)
                        if isinstance(dest, (bytes, str, PSLiteral)):
                            dest_name = (
                                dest.name
                                if isinstance(dest, PSLiteral)
                                else (
                                    dest.decode("utf-8", errors="ignore")
                                    if isinstance(dest, bytes)
                                    else dest
                                )
                            )
                            try:
                                dest = resolve1(pdf.doc.get_dest(dest_name))
                            except Exception:
                                return None
                        if isinstance(dest, list) and len(dest) > 0:
                            page_ref = dest[0]
                            if isinstance(page_ref, PDFObjRef):
                                page_ref = resolve1(page_ref)
                            if hasattr(page_ref, "objid"):
                                objid = page_ref.objid
                                if objid in page_id_to_num:
                                    return page_id_to_num[objid]
                            page_id = id(page_ref)
                            if page_id in page_id_to_num:
                                return page_id_to_num[page_id]
                            if hasattr(page_ref, "objid"):
                                target_objid = getattr(page_ref, "objid", None)
                                for pg_num, pg in enumerate(pdf.pages, start=1):
                                    page_obj = getattr(pg, "page_obj", None)
                                    if (
                                        page_obj
                                        and getattr(page_obj, "objid", None)
                                        == target_objid
                                    ):
                                        return pg_num
                        if isinstance(dest, dict) and "D" in dest:
                            return resolve_dest_to_page(dest["D"])
                    except Exception:
                        pass
                    return None

                # Extract embedded PDF outline/bookmarks
                try:
                    outlines = pdf.doc.get_outlines()
                    for level, title, dest, action, _se in outlines:
                        if isinstance(title, bytes):
                            title = title.decode("utf-8", errors="ignore")
                        title = str(title).strip() if title else ""
                        if not title:
                            continue
                        page_num = None
                        if dest:
                            page_num = resolve_dest_to_page(dest)
                        if page_num is None and action:
                            try:
                                action_resolved = resolve1(action)
                                if (
                                    isinstance(action_resolved, dict)
                                    and "D" in action_resolved
                                ):
                                    page_num = resolve_dest_to_page(
                                        action_resolved["D"]
                                    )
                            except Exception:
                                pass
                        if page_num is None:
                            page_num = 1
                        metadata["outline"].append(
                            {"level": level, "title": title[:200], "page": page_num}
                        )
                        outline_extracted = True
                except PDFNoOutlines:
                    pass
                except Exception as e:
                    logger.debug("Could not extract PDF outline: %s", e)

            # Single iteration through all pages for positions AND summaries
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract word positions
                words = page.extract_words(use_text_flow=True)
                if words:
                    for word in words:
                        text_positions.append(
                            {
                                "text": word.get("text", ""),
                                "page": page_num,
                                "x0": word.get("x0", 0),
                                "top": word.get("top", 0),
                                "x1": word.get("x1", 0),
                                "bottom": word.get("bottom", 0),
                            }
                        )

                # Extract page text
                page_text = page.extract_text() or ""
                if page_text:
                    document_text += page_text + "\n\n"

                # Build page summary if extracting metadata
                if extract_metadata and metadata is not None:
                    first_line = ""
                    char_count = len(page_text)
                    if page_text:
                        lines = page_text.strip().split("\n")
                        for line in lines:
                            stripped = line.strip()
                            if (
                                stripped
                                and len(stripped) > 5
                                and not stripped.isdigit()
                                and not stripped.lower().startswith("page ")
                            ):
                                first_line = stripped[:150]
                                break
                    metadata["page_summaries"].append(
                        {
                            "page": page_num,
                            "first_line": first_line,
                            "char_count": char_count,
                        }
                    )

            # Heuristic outline detection if no embedded outline
            if extract_metadata and metadata is not None and not outline_extracted:
                for summary in metadata["page_summaries"]:
                    first_line = summary.get("first_line", "")
                    if not first_line:
                        continue
                    is_section = (
                        first_line.isupper()
                        or re.match(r"^\d+\.\s+\w", first_line)
                        or re.match(r"^[IVXLCDM]+\.\s+\w", first_line)
                        or re.match(
                            r"^(EXHIBIT|SECTION|ARTICLE|ITEM)\s+", first_line, re.I
                        )
                        or first_line.endswith(":")
                    )
                    if is_section:
                        metadata["outline"].append(
                            {
                                "level": 1,
                                "title": first_line[:100],
                                "page": summary["page"],
                            }
                        )

        return document_text, text_positions, metadata

    @staticmethod
    def extract_pdf_metadata(pdf_bytes: bytes) -> dict:
        """Extract PDF metadata including outline/TOC and page summaries.

        Convenience wrapper around extract_pdf_with_positions for metadata-only extraction.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF file bytes

        Returns
        -------
        dict
            Metadata dict with keys:
            - 'page_count': int
            - 'outline': list of {level, title, page} dicts (table of contents)
            - 'page_summaries': list of {page, first_line, char_count} dicts
            - 'info': dict of PDF info (title, author, etc.)
        """
        _, _, metadata = DocumentProcessor.extract_pdf_with_positions(
            pdf_bytes, extract_metadata=True
        )
        return metadata or {
            "page_count": 0,
            "outline": [],
            "page_summaries": [],
            "info": {},
        }

    async def store_pdf_positions_in_snowflake(
        self,
        client,
        file_name: str,
        stage_path: str,
        positions: list[dict],
    ) -> bool:
        """Store PDF text positions in Snowflake for citation highlighting.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_name : str
            The PDF filename
        stage_path : str
            The stage path where PDF is stored
        positions : list[dict]
            List of position dictionaries with text and coordinates

        Returns
        -------
        bool
            True if storage successful, False otherwise
        """
        if not positions:
            return False

        user_schema = await self._get_user_schema(client)
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS OPENBB_AGENTS.{user_schema}.PDF_TEXT_POSITIONS (
            FILE_NAME VARCHAR(16777216),
            STAGE_PATH VARCHAR(16777216),
            PAGE INTEGER,
            TEXT VARCHAR(16777216),
            X0 FLOAT,
            TOP FLOAT,
            X1 FLOAT,
            BOTTOM FLOAT,
            EXTRACTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """
        await self._run_in_thread(client.execute_statement, create_table_sql)

        delete_sql = f"""
        DELETE FROM OPENBB_AGENTS.{user_schema}.PDF_TEXT_POSITIONS
        WHERE FILE_NAME = '{file_name}' OR STAGE_PATH = '{stage_path}'
        """
        await self._run_in_thread(client.execute_statement, delete_sql)

        batch_size = 100
        for i in range(0, len(positions), batch_size):
            batch = positions[i : i + batch_size]
            values = []
            for pos in batch:
                text_escaped = pos["text"].replace("'", "''")
                stage_path_escaped = stage_path.replace("'", "''")
                file_name_escaped = file_name.replace("'", "''")
                values.append(
                    f"('{file_name_escaped}', '{stage_path_escaped}', "
                    f"{pos['page']}, '{text_escaped}', "
                    f"{pos['x0']}, {pos['top']}, {pos['x1']}, {pos['bottom']})"
                )
            if values:
                insert_sql = f"""
                INSERT INTO OPENBB_AGENTS.{user_schema}.PDF_TEXT_POSITIONS
                (FILE_NAME, STAGE_PATH, PAGE, TEXT, X0, TOP, X1, BOTTOM)
                VALUES {','.join(values)}
                """
                await self._run_in_thread(client.execute_statement, insert_sql)

        return True

    # ------------------------------------------------------------------
    # Embedding infrastructure
    # ------------------------------------------------------------------

    async def _create_document_embeddings_table(self, client) -> bool:
        """Create DOCUMENT_EMBEDDINGS table with vector index.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        bool
            True if table and index created successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS"

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                EMBEDDING_ID STRING PRIMARY KEY,
                FILE_NAME STRING NOT NULL,
                STAGE_PATH STRING,
                PAGE_NUMBER INTEGER,
                CHUNK_INDEX INTEGER,
                CONTENT_TYPE STRING NOT NULL,
                CHUNK_TEXT STRING,
                IMAGE_STAGE_PATH STRING,
                IMAGE_HASH STRING,
                EMBEDDING VECTOR(FLOAT, 1024) NOT NULL,
                EMBEDDING_MODEL STRING DEFAULT 'snowflake-arctic-embed-l-v2.0',
                CHUNK_START_CHAR INTEGER,
                CHUNK_END_CHAR INTEGER,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            await self._run_in_thread(client.execute_statement, create_table_sql)
            logger.info("Created DOCUMENT_EMBEDDINGS table for %s", user_schema)

            return True
        except Exception as e:
            logger.error("Failed to create embeddings table: %s", e, exc_info=True)
            return False

    async def _create_processing_jobs_table(self, client) -> bool:
        """Create DOCUMENT_PROCESSING_JOBS table for tracking background job status.

        This table tracks the progress of document processing jobs including
        Cortex AI parsing and embedding generation. Each job uses Snowflake's
        SQLID tracking to monitor the query chain.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        bool
            True if table created successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PROCESSING_JOBS"

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                JOB_ID STRING PRIMARY KEY,
                FILE_NAME STRING NOT NULL,
                STAGE_PATH STRING NOT NULL,
                STATUS STRING NOT NULL DEFAULT 'pending',
                LAST_COMPLETED_STEP STRING,
                PARSE_QUERY_ID STRING,
                EMBED_QUERY_IDS ARRAY,
                PAGE_COUNT INTEGER DEFAULT 0,
                EMBEDDING_COUNT INTEGER DEFAULT 0,
                ERROR_MESSAGE STRING,
                EMBED_IMAGES BOOLEAN DEFAULT FALSE,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            COMMENT = 'Tracks document processing jobs for Cortex parsing and embedding generation'
            """
            await self._run_in_thread(client.execute_statement, create_table_sql)
            logger.info("Created DOCUMENT_PROCESSING_JOBS table for %s", user_schema)

            return True
        except Exception as e:
            logger.error("Failed to create processing jobs table: %s", e, exc_info=True)
            return False

    async def create_processing_job(
        self,
        client,
        file_name: str,
        stage_path: str,
        embed_images: bool = False,
    ) -> str | None:
        """Create a new document processing job record.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_name : str
            Name of the file being processed
        stage_path : str
            Stage path where the file is stored
        embed_images : bool
            Whether to embed images during processing

        Returns
        -------
        str | None
            Job ID if created successfully, None otherwise
        """
        import uuid

        try:
            await self._create_processing_jobs_table(client)
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PROCESSING_JOBS"

            job_id = str(uuid.uuid4())
            escaped_filename = file_name.replace("'", "''")
            escaped_stage_path = stage_path.replace("'", "''")

            insert_sql = f"""
            INSERT INTO {qualified_table} (
                JOB_ID, FILE_NAME, STAGE_PATH, STATUS, EMBED_IMAGES
            ) VALUES (
                '{job_id}',
                '{escaped_filename}',
                '{escaped_stage_path}',
                'pending',
                {str(embed_images).upper()}
            )
            """
            await self._run_in_thread(client.execute_statement, insert_sql)
            logger.info("Created processing job %s for %s", job_id, file_name)

            return job_id
        except Exception as e:
            logger.error("Failed to create processing job: %s", e, exc_info=True)
            return None

    async def update_processing_job(
        self,
        client,
        job_id: str,
        status: str | None = None,
        last_completed_step: str | None = None,
        parse_query_id: str | None = None,
        embed_query_id: str | None = None,
        page_count: int | None = None,
        embedding_count: int | None = None,
        error_message: str | None = None,
    ) -> bool:
        """Update a document processing job record.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        job_id : str
            The job ID to update
        status : str | None
            New status (pending, parsing, embedding, completed, failed)
        last_completed_step : str | None
            Name of the last completed processing step
        parse_query_id : str | None
            Query ID from the parse operation
        embed_query_id : str | None
            Query ID to add to the embed_query_ids array
        page_count : int | None
            Number of pages parsed
        embedding_count : int | None
            Number of embeddings generated
        error_message : str | None
            Error message if job failed

        Returns
        -------
        bool
            True if updated successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PROCESSING_JOBS"

            updates = ["UPDATED_AT = CURRENT_TIMESTAMP()"]

            if status is not None:
                updates.append(f"STATUS = '{status}'")
            if last_completed_step is not None:
                updates.append(f"LAST_COMPLETED_STEP = '{last_completed_step}'")
            if parse_query_id is not None:
                updates.append(f"PARSE_QUERY_ID = '{parse_query_id}'")
            if embed_query_id is not None:
                updates.append(
                    f"EMBED_QUERY_IDS = ARRAY_APPEND(COALESCE(EMBED_QUERY_IDS, ARRAY_CONSTRUCT()), '{embed_query_id}')"
                )
            if page_count is not None:
                updates.append(f"PAGE_COUNT = {page_count}")
            if embedding_count is not None:
                updates.append(f"EMBEDDING_COUNT = {embedding_count}")
            if error_message is not None:
                escaped_error = error_message.replace("'", "''")[:1000]  # Limit length
                updates.append(f"ERROR_MESSAGE = '{escaped_error}'")

            update_sql = f"""
            UPDATE {qualified_table}
            SET {', '.join(updates)}
            WHERE JOB_ID = '{job_id}'
            """
            await self._run_in_thread(client.execute_statement, update_sql)
            return True
        except Exception as e:
            logger.error("Failed to update processing job %s: %s", job_id, e)
            return False

    async def get_processing_job_status(
        self,
        client,
        job_id: str,
    ) -> dict | None:
        """Get the status of a document processing job.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        job_id : str
            The job ID to query

        Returns
        -------
        dict | None
            Job status dict with all fields, or None if not found
        """
        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PROCESSING_JOBS"

            query_sql = f"""
            SELECT 
                JOB_ID,
                FILE_NAME,
                STAGE_PATH,
                STATUS,
                LAST_COMPLETED_STEP,
                PARSE_QUERY_ID,
                EMBED_QUERY_IDS,
                PAGE_COUNT,
                EMBEDDING_COUNT,
                ERROR_MESSAGE,
                EMBED_IMAGES,
                CREATED_AT,
                UPDATED_AT
            FROM {qualified_table}
            WHERE JOB_ID = '{job_id}'
            """
            result_json = await self._run_in_thread(client.execute_query, query_sql)
            result = json.loads(result_json) if result_json else {}

            rows = result.get("rowData", [])
            if not rows:
                return None

            row = rows[0]
            return {
                "job_id": self._get_row_value(row, "JOB_ID"),
                "file_name": self._get_row_value(row, "FILE_NAME"),
                "stage_path": self._get_row_value(row, "STAGE_PATH"),
                "status": self._get_row_value(row, "STATUS"),
                "last_completed_step": self._get_row_value(row, "LAST_COMPLETED_STEP"),
                "parse_query_id": self._get_row_value(row, "PARSE_QUERY_ID"),
                "embed_query_ids": self._get_row_value(row, "EMBED_QUERY_IDS"),
                "page_count": self._get_row_value(row, "PAGE_COUNT"),
                "embedding_count": self._get_row_value(row, "EMBEDDING_COUNT"),
                "error_message": self._get_row_value(row, "ERROR_MESSAGE"),
                "embed_images": self._get_row_value(row, "EMBED_IMAGES"),
                "created_at": self._get_row_value(row, "CREATED_AT"),
                "updated_at": self._get_row_value(row, "UPDATED_AT"),
            }
        except Exception as e:
            logger.error("Failed to get processing job status: %s", e)
            return None

    async def check_document_ready(
        self,
        client,
        stage_path: str,
    ) -> dict:
        """Check what data is available for a document.

        Checks if positions, metadata, parsed pages, and embeddings exist
        for the given document. This is used by AI chat to determine what
        context can be provided.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        stage_path : str
            Stage path of the document to check

        Returns
        -------
        dict
            Dict with availability status:
            - has_positions: bool
            - has_metadata: bool
            - has_parsed_pages: bool
            - has_embeddings: bool
            - page_count: int
            - embedding_count: int
            - processing_status: str | None (from job table if exists)
        """
        try:
            user_schema = await self._get_user_schema(client)
            escaped_path = stage_path.replace("'", "''")

            # Check positions
            positions_table = f"OPENBB_AGENTS.{user_schema}.PDF_TEXT_POSITIONS"
            positions_sql = f"""
            SELECT COUNT(*) as cnt 
            FROM {positions_table} 
            WHERE STAGE_PATH = '{escaped_path}'
            """
            try:
                result = await self._run_in_thread(client.execute_query, positions_sql)
                data = json.loads(result) if result else {}
                has_positions = (
                    self._get_row_value(
                        data.get("rowData", [{}])[0], "CNT", "cnt", default=0
                    )
                    > 0
                )
            except Exception:
                has_positions = False

            # Check metadata
            metadata_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_METADATA"
            metadata_sql = f"""
            SELECT COUNT(*) as cnt 
            FROM {metadata_table} 
            WHERE STAGE_PATH = '{escaped_path}'
            """
            try:
                result = await self._run_in_thread(client.execute_query, metadata_sql)
                data = json.loads(result) if result else {}
                has_metadata = (
                    self._get_row_value(
                        data.get("rowData", [{}])[0], "CNT", "cnt", default=0
                    )
                    > 0
                )
            except Exception:
                has_metadata = False

            # Check parsed pages
            pages_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS"
            pages_sql = f"""
            SELECT COUNT(*) as cnt 
            FROM {pages_table} 
            WHERE STAGE_PATH = '{escaped_path}'
            """
            try:
                result = await self._run_in_thread(client.execute_query, pages_sql)
                data = json.loads(result) if result else {}
                page_count = self._get_row_value(
                    data.get("rowData", [{}])[0], "CNT", "cnt", default=0
                )
                has_parsed_pages = page_count > 0
            except Exception:
                has_parsed_pages = False
                page_count = 0

            # Check embeddings
            embeddings_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS"
            embeddings_sql = f"""
            SELECT COUNT(*) as cnt 
            FROM {embeddings_table} 
            WHERE STAGE_PATH = '{escaped_path}'
            """
            try:
                result = await self._run_in_thread(client.execute_query, embeddings_sql)
                data = json.loads(result) if result else {}
                embedding_count = self._get_row_value(
                    data.get("rowData", [{}])[0], "CNT", "cnt", default=0
                )
                has_embeddings = embedding_count > 0
            except Exception:
                has_embeddings = False
                embedding_count = 0

            # Check processing job status (most recent job for this file)
            jobs_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PROCESSING_JOBS"
            jobs_sql = f"""
            SELECT STATUS 
            FROM {jobs_table} 
            WHERE STAGE_PATH = '{escaped_path}'
            ORDER BY CREATED_AT DESC
            LIMIT 1
            """
            try:
                result = await self._run_in_thread(client.execute_query, jobs_sql)
                data = json.loads(result) if result else {}
                rows = data.get("rowData", [])
                processing_status = (
                    self._get_row_value(rows[0], "STATUS") if rows else None
                )
            except Exception:
                processing_status = None

            return {
                "has_positions": has_positions,
                "has_metadata": has_metadata,
                "has_parsed_pages": has_parsed_pages,
                "has_embeddings": has_embeddings,
                "page_count": page_count,
                "embedding_count": embedding_count,
                "processing_status": processing_status,
            }
        except Exception as e:
            logger.error("Failed to check document ready status: %s", e)
            return {
                "has_positions": False,
                "has_metadata": False,
                "has_parsed_pages": False,
                "has_embeddings": False,
                "page_count": 0,
                "embedding_count": 0,
                "processing_status": None,
            }

    async def _create_process_document_procedure(self, client) -> bool:
        """Create Snowflake stored procedure for async document processing.

        This procedure handles Cortex AI parsing and embedding generation
        using Snowflake Scripting. It uses SQLID to track each query in the
        processing chain for status monitoring.

        The procedure:
        1. Updates job status to 'parsing'
        2. Deletes any existing data for this document (prevents duplicates)
        3. Parses document with AI_PARSE_DOCUMENT
        4. Captures parse query ID via SQLID
        5. Updates job status to 'embedding'
        6. Generates text embeddings for each page chunk
        7. If embed_images=TRUE, generates image embeddings from DOCUMENT_IMAGES_METADATA
        8. Updates job status to 'completed' or 'failed'

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        bool
            True if procedure created successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            db_name = "OPENBB_AGENTS"
            qualified_proc = f"{db_name}.{user_schema}.PROCESS_DOCUMENT_ASYNC"
            jobs_table = f"{db_name}.{user_schema}.DOCUMENT_PROCESSING_JOBS"
            parse_table = f"{db_name}.{user_schema}.DOCUMENT_PARSE_RESULTS"
            embed_table = f"{db_name}.{user_schema}.DOCUMENT_EMBEDDINGS"
            images_meta_table = f"{db_name}.{user_schema}.DOCUMENT_IMAGES_METADATA"
            images_stage = f"{db_name}.{user_schema}.DOCUMENT_IMAGES"

            # Create the stored procedure using Snowflake Scripting
            create_proc_sql = f"""
            CREATE OR REPLACE PROCEDURE {qualified_proc}(
                P_JOB_ID VARCHAR,
                P_FILE_NAME VARCHAR,
                P_STAGE_PATH VARCHAR,
                P_EMBED_IMAGES BOOLEAN DEFAULT FALSE
            )
            RETURNS VARCHAR
            LANGUAGE SQL
            EXECUTE AS CALLER
            AS
            $$
            DECLARE
                v_stage_name VARCHAR;
                v_file_path VARCHAR;
                v_parse_query_id VARCHAR;
                v_page_count INTEGER DEFAULT 0;
                v_text_embedding_count INTEGER DEFAULT 0;
                v_image_embedding_count INTEGER DEFAULT 0;
                v_total_embedding_count INTEGER DEFAULT 0;
                v_error_msg VARCHAR;
                v_clean_path VARCHAR;
                v_insert_sql VARCHAR;
            BEGIN
                -- Update job status to parsing
                UPDATE {jobs_table}
                SET STATUS = 'parsing', UPDATED_AT = CURRENT_TIMESTAMP()
                WHERE JOB_ID = :P_JOB_ID;

                -- Extract stage info from stage_path
                v_clean_path := LTRIM(:P_STAGE_PATH, '@');
                IF (POSITION('/' IN v_clean_path) > 0) THEN
                    v_stage_name := '@' || SPLIT_PART(v_clean_path, '/', 1);
                    v_file_path := SUBSTR(v_clean_path, POSITION('/' IN v_clean_path) + 1);
                ELSE
                    v_stage_name := '@' || v_clean_path;
                    v_file_path := :P_FILE_NAME;
                END IF;

                -- Create parse results table if not exists
                CREATE TABLE IF NOT EXISTS {parse_table} (
                    FILE_NAME STRING,
                    STAGE_PATH STRING,
                    PAGE_NUMBER INTEGER,
                    PAGE_CONTENT STRING,
                    METADATA VARIANT,
                    PARSED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
                );

                -- Delete existing parse results for this document (prevent duplicates on re-upload)
                DELETE FROM {parse_table}
                WHERE FILE_NAME = :P_FILE_NAME OR STAGE_PATH = :P_STAGE_PATH;

                -- Parse document with AI_PARSE_DOCUMENT using dynamic SQL
                BEGIN
                    v_insert_sql := '
                        INSERT INTO {parse_table} (FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, METADATA)
                        WITH DOC AS (
                            SELECT AI_PARSE_DOCUMENT(
                                TO_FILE(''' || v_stage_name || ''', ''' || v_file_path || '''), 
                                {{''mode'': ''LAYOUT'', ''page_split'': true}}
                            ) AS RAW
                        )
                        SELECT 
                            ''' || REPLACE(:P_FILE_NAME, '''', '''''') || ''',
                            ''' || REPLACE(:P_STAGE_PATH, '''', '''''') || ''',
                            INDEX + 1,
                            GET(VALUE, ''content'')::STRING,
                            GET(DOC.RAW, ''metadata'')
                        FROM DOC, TABLE(FLATTEN(input => COALESCE(GET(DOC.RAW, ''pages''), DOC.RAW)))
                    ';
                    EXECUTE IMMEDIATE v_insert_sql;

                    -- Capture the query ID from the insert
                    v_parse_query_id := LAST_QUERY_ID();

                    -- Get page count
                    SELECT COUNT(*) INTO v_page_count
                    FROM {parse_table}
                    WHERE FILE_NAME = :P_FILE_NAME AND STAGE_PATH = :P_STAGE_PATH;

                    -- Update job with parse info
                    UPDATE {jobs_table}
                    SET 
                        PARSE_QUERY_ID = :v_parse_query_id,
                        PAGE_COUNT = :v_page_count,
                        LAST_COMPLETED_STEP = 'parse',
                        STATUS = 'embedding',
                        UPDATED_AT = CURRENT_TIMESTAMP()
                    WHERE JOB_ID = :P_JOB_ID;

                EXCEPTION
                    WHEN OTHER THEN
                        v_error_msg := 'Parse failed: ' || SQLERRM;
                        UPDATE {jobs_table}
                        SET STATUS = 'failed', ERROR_MESSAGE = :v_error_msg, UPDATED_AT = CURRENT_TIMESTAMP()
                        WHERE JOB_ID = :P_JOB_ID;
                        RETURN v_error_msg;
                END;

                -- Create embeddings table if not exists
                CREATE TABLE IF NOT EXISTS {embed_table} (
                    EMBEDDING_ID STRING PRIMARY KEY,
                    FILE_NAME STRING NOT NULL,
                    STAGE_PATH STRING,
                    PAGE_NUMBER INTEGER,
                    CHUNK_INDEX INTEGER,
                    CONTENT_TYPE STRING NOT NULL,
                    CHUNK_TEXT STRING,
                    IMAGE_STAGE_PATH STRING,
                    IMAGE_HASH STRING,
                    EMBEDDING VECTOR(FLOAT, 1024) NOT NULL,
                    EMBEDDING_MODEL STRING DEFAULT 'snowflake-arctic-embed-l-v2.0',
                    CHUNK_START_CHAR INTEGER,
                    CHUNK_END_CHAR INTEGER,
                    CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
                );

                -- Delete existing embeddings for this document (prevent duplicates on re-upload)
                DELETE FROM {embed_table}
                WHERE FILE_NAME = :P_FILE_NAME;

                -- Update status: starting text embeddings
                UPDATE {jobs_table}
                SET STATUS = 'embedding_text', UPDATED_AT = CURRENT_TIMESTAMP()
                WHERE JOB_ID = :P_JOB_ID;

                -- Generate TEXT embeddings for each page
                BEGIN
                    INSERT INTO {embed_table} (
                        EMBEDDING_ID, FILE_NAME, STAGE_PATH, PAGE_NUMBER, 
                        CHUNK_INDEX, CONTENT_TYPE, CHUNK_TEXT, EMBEDDING, EMBEDDING_MODEL
                    )
                    SELECT 
                        UUID_STRING(),
                        FILE_NAME,
                        STAGE_PATH,
                        PAGE_NUMBER,
                        0,  -- chunk_index (whole page for now)
                        'text',
                        PAGE_CONTENT,
                        SNOWFLAKE.CORTEX.EMBED_TEXT_1024('snowflake-arctic-embed-l-v2.0', PAGE_CONTENT),
                        'snowflake-arctic-embed-l-v2.0'
                    FROM {parse_table}
                    WHERE FILE_NAME = :P_FILE_NAME 
                      AND STAGE_PATH = :P_STAGE_PATH
                      AND PAGE_CONTENT IS NOT NULL
                      AND LENGTH(TRIM(PAGE_CONTENT)) > 10;

                    -- Get text embedding count
                    SELECT COUNT(*) INTO v_text_embedding_count
                    FROM {embed_table}
                    WHERE FILE_NAME = :P_FILE_NAME AND CONTENT_TYPE = 'text';

                EXCEPTION
                    WHEN OTHER THEN
                        v_error_msg := 'Text embedding failed: ' || SQLERRM;
                        UPDATE {jobs_table}
                        SET STATUS = 'failed', ERROR_MESSAGE = :v_error_msg, UPDATED_AT = CURRENT_TIMESTAMP()
                        WHERE JOB_ID = :P_JOB_ID;
                        RETURN v_error_msg;
                END;

                -- Generate IMAGE embeddings if enabled and images exist in metadata table
                IF (:P_EMBED_IMAGES = TRUE) THEN
                    -- Update status: starting image embeddings
                    UPDATE {jobs_table}
                    SET STATUS = 'embedding_images', UPDATED_AT = CURRENT_TIMESTAMP()
                    WHERE JOB_ID = :P_JOB_ID;

                    BEGIN
                        -- Check if images metadata table exists and has images for this document
                        -- Images were uploaded by the client before calling this procedure
                        INSERT INTO {embed_table} (
                            EMBEDDING_ID, FILE_NAME, STAGE_PATH, PAGE_NUMBER, 
                            CHUNK_INDEX, CONTENT_TYPE, CHUNK_TEXT, IMAGE_STAGE_PATH, IMAGE_HASH,
                            EMBEDDING, EMBEDDING_MODEL
                        )
                        SELECT 
                            UUID_STRING(),
                            im.FILE_NAME,
                            im.STAGE_PATH,
                            im.PAGE_NUMBER,
                            im.IMAGE_INDEX,
                            'image',
                            COALESCE(pr.PAGE_CONTENT, '')::VARCHAR,  -- Link to page text for context
                            im.IMAGE_STAGE_PATH,
                            im.IMAGE_HASH,
                            AI_EMBED(
                                'voyage-multimodal-3',
                                TO_FILE('@{images_stage}', im.RELATIVE_PATH)
                            ),
                            'voyage-multimodal-3'
                        FROM {images_meta_table} im
                        LEFT JOIN {parse_table} pr 
                            ON im.FILE_NAME = pr.FILE_NAME AND im.PAGE_NUMBER = pr.PAGE_NUMBER
                        WHERE im.FILE_NAME = :P_FILE_NAME;

                        -- Get image embedding count
                        SELECT COUNT(*) INTO v_image_embedding_count
                        FROM {embed_table}
                        WHERE FILE_NAME = :P_FILE_NAME AND CONTENT_TYPE = 'image';

                    EXCEPTION
                        WHEN OTHER THEN
                            -- Image embedding failed - update job with error but continue
                            v_image_embedding_count := 0;
                            UPDATE {jobs_table}
                            SET ERROR_MESSAGE = 'Image embedding failed: ' || SQLERRM, UPDATED_AT = CURRENT_TIMESTAMP()
                            WHERE JOB_ID = :P_JOB_ID;
                    END;
                END IF;

                -- Calculate total embeddings
                v_total_embedding_count := v_text_embedding_count + v_image_embedding_count;

                -- Update job with final embedding info
                UPDATE {jobs_table}
                SET 
                    EMBED_QUERY_IDS = ARRAY_CONSTRUCT(LAST_QUERY_ID()),
                    EMBEDDING_COUNT = :v_total_embedding_count,
                    LAST_COMPLETED_STEP = 'embed',
                    STATUS = 'completed',
                    UPDATED_AT = CURRENT_TIMESTAMP()
                WHERE JOB_ID = :P_JOB_ID;

                RETURN 'completed: ' || :v_text_embedding_count || ' text, ' || :v_image_embedding_count || ' images';
            END;
            $$;
            """
            await self._run_in_thread(client.execute_statement, create_proc_sql)
            logger.info("Created PROCESS_DOCUMENT_ASYNC procedure for %s", user_schema)

            return True
        except Exception as e:
            logger.error(
                "Failed to create process document procedure: %s", e, exc_info=True
            )
            return False

    async def start_document_processing_async(
        self,
        client,
        job_id: str,
        file_name: str,
        stage_path: str,
        embed_images: bool = False,
    ) -> bool:
        """Start async document processing by calling the stored procedure.

        This starts the processing in the background without blocking.
        The job status can be monitored via get_processing_job_status().

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        job_id : str
            The job ID to track
        file_name : str
            Name of the file being processed
        stage_path : str
            Stage path where the file is stored
        embed_images : bool
            Whether to embed images

        Returns
        -------
        bool
            True if processing started successfully
        """
        try:
            # Ensure procedure exists
            await self._create_process_document_procedure(client)

            user_schema = await self._get_user_schema(client)
            db_name = "OPENBB_AGENTS"
            qualified_proc = f"{db_name}.{user_schema}.PROCESS_DOCUMENT_ASYNC"

            # Escape parameters
            escaped_job_id = job_id.replace("'", "''")
            escaped_filename = file_name.replace("'", "''")
            escaped_stage_path = stage_path.replace("'", "''")

            # Call procedure (this runs asynchronously in Snowflake)
            call_sql = f"""
            CALL {qualified_proc}(
                '{escaped_job_id}',
                '{escaped_filename}',
                '{escaped_stage_path}',
                {str(embed_images).upper()}
            )
            """

            logger.info(
                "[Job %s] Starting document processing for %s (embed_images=%s)",
                job_id,
                file_name,
                embed_images,
            )

            # Execute the stored procedure (this blocks until complete)
            await self._run_in_thread(client.execute_statement, call_sql)

            # Monitor job status with polling loop
            max_wait_seconds = 900  # 15 minute timeout
            poll_interval = 2  # Check every 2 seconds
            elapsed = 0
            last_status = None
            last_step = None

            while elapsed < max_wait_seconds:
                job_info = await self.get_processing_job_status(client, job_id)
                if not job_info:
                    logger.warning("[Job %s] Job not found in status table", job_id)
                    break

                current_status = job_info.get("status")
                current_step = job_info.get("last_completed_step")
                error_msg = job_info.get("error_message")

                # Log status transitions
                if current_status != last_status or current_step != last_step:
                    if current_status == "failed":
                        logger.error(
                            "[Job %s] FAILED at step '%s': %s",
                            job_id,
                            current_step or "unknown",
                            error_msg or "No error message",
                        )
                    else:
                        logger.info(
                            "[Job %s] Status: %s, Step: %s",
                            job_id,
                            current_status,
                            current_step or "starting",
                        )
                    last_status = current_status
                    last_step = current_step

                # Check for terminal states
                if current_status in ("completed", "failed"):
                    if current_status == "completed":
                        page_count = job_info.get("page_count", 0)
                        embed_count = job_info.get("embedding_count", 0)
                        logger.info(
                            "[Job %s] COMPLETED: %d pages, %d embeddings",
                            job_id,
                            page_count,
                            embed_count,
                        )
                    return current_status == "completed"

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            # Timeout reached
            logger.error(
                "[Job %s] TIMEOUT after %d seconds - marking as failed",
                job_id,
                max_wait_seconds,
            )
            await self.update_processing_job(
                client,
                job_id,
                status="failed",
                error_message=f"Processing timed out after {max_wait_seconds} seconds",
            )
            return False
        except Exception as e:
            logger.error("Failed to start document processing: %s", e, exc_info=True)
            return False

    async def _create_document_images_stage(self, client) -> bool:
        """Create stage for document image storage.

        Images are organized simply: @STAGE/filename/page_X_image_Y.format

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        bool
            True if stage created successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            db_name = "OPENBB_AGENTS"
            stage_name = "DOCUMENT_IMAGES"
            qualified_stage = f'"{db_name}"."{user_schema}"."{stage_name}"'

            create_stage_sql = f"""
            CREATE STAGE IF NOT EXISTS {qualified_stage}
            ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
            DIRECTORY = (ENABLE = TRUE)
            """
            await self._run_in_thread(client.execute_statement, create_stage_sql)
            logger.debug("Created DOCUMENT_IMAGES stage for %s", user_schema)
            return True
        except Exception as e:
            logger.error("Failed to create document images stage: %s", e, exc_info=True)
            return False

    async def _create_document_images_metadata_table(self, client) -> bool:
        """Create table for tracking uploaded document images.

        This table stores metadata about images extracted from PDFs and uploaded
        to the DOCUMENT_IMAGES stage. The stored procedure uses this table to
        generate image embeddings.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        bool
            True if table created successfully
        """
        try:
            user_schema = await self._get_user_schema(client)
            db_name = "OPENBB_AGENTS"
            table_name = f"{db_name}.{user_schema}.DOCUMENT_IMAGES_METADATA"

            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                ID STRING PRIMARY KEY DEFAULT UUID_STRING(),
                FILE_NAME STRING NOT NULL,
                STAGE_PATH STRING,
                PAGE_NUMBER INTEGER NOT NULL,
                IMAGE_INDEX INTEGER NOT NULL,
                RELATIVE_PATH STRING NOT NULL,
                IMAGE_STAGE_PATH STRING NOT NULL,
                IMAGE_HASH STRING,
                IMAGE_FORMAT STRING,
                WIDTH INTEGER,
                HEIGHT INTEGER,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            await self._run_in_thread(client.execute_statement, create_table_sql)
            logger.debug("Created DOCUMENT_IMAGES_METADATA table for %s", user_schema)
            return True
        except Exception as e:
            logger.error(
                "Failed to create document images metadata table: %s", e, exc_info=True
            )
            return False

    async def upload_images_and_store_metadata(
        self,
        client,
        file_name: str,
        stage_path: str,
        pdf_bytes: bytes,
    ) -> int:
        """Extract images from PDF, upload to stage, and store metadata.

        This is called BEFORE the stored procedure runs. The procedure will
        read from DOCUMENT_IMAGES_METADATA to generate image embeddings.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_name : str
            The document filename
        stage_path : str
            The stage path where document is stored
        pdf_bytes : bytes
            Raw PDF bytes for image extraction

        Returns
        -------
        int
            Number of images uploaded and stored
        """
        try:
            logger.info("Starting image extraction and upload for %s", file_name)

            # Ensure stage and table exist
            await self._create_document_images_stage(client)
            await self._create_document_images_metadata_table(client)

            user_schema = await self._get_user_schema(client)
            db_name = "OPENBB_AGENTS"
            stage_name = "DOCUMENT_IMAGES"
            qualified_stage = f"{db_name}.{user_schema}.{stage_name}"
            metadata_table = f"{db_name}.{user_schema}.DOCUMENT_IMAGES_METADATA"

            # Clean up old images for this document
            file_name_escaped = file_name.replace("'", "''")
            try:
                # Delete from metadata table
                delete_sql = f"DELETE FROM {metadata_table} WHERE FILE_NAME = '{file_name_escaped}'"
                await self._run_in_thread(client.execute_statement, delete_sql)
                # Remove from stage
                remove_sql = f"REMOVE @{qualified_stage}/{file_name_escaped}/"
                await self._run_in_thread(client.execute_statement, remove_sql)
                logger.info("Cleaned up old images for %s", file_name)
            except Exception as e:
                logger.debug("Stage/table cleanup (may be empty): %s", e)

            # Extract images from PDF
            images = self._extract_images_from_pdf(pdf_bytes)
            logger.info("Extracted %d images from %s", len(images), file_name)

            if not images:
                return 0

            # Upload images CONCURRENTLY with a semaphore to limit parallelism
            upload_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent uploads
            uploaded_count = 0
            upload_lock = asyncio.Lock()

            async def upload_single_image(img):
                nonlocal uploaded_count
                async with upload_semaphore:
                    try:
                        # Simple path: filename/page_X_image_Y.format
                        relative_path = f"{file_name}/page_{img['page_number']}_image_{img['image_index']}.{img['format']}"
                        full_stage_path = f"@{qualified_stage}/{relative_path}"

                        # Upload image bytes to stage
                        await self._run_in_thread(
                            client.upload_bytes_to_stage,
                            img["image_bytes"],
                            relative_path,
                            stage_name,
                        )

                        # Store metadata in table
                        stage_path_escaped = stage_path.replace("'", "''")
                        relative_path_escaped = relative_path.replace("'", "''")
                        full_stage_path_escaped = full_stage_path.replace("'", "''")
                        img_hash = img.get("image_hash", "")
                        img_format = img.get("format", "jpeg")
                        width = img.get("width", 0)
                        height = img.get("height", 0)

                        insert_sql = f"""
                        INSERT INTO {metadata_table} (
                            FILE_NAME, STAGE_PATH, PAGE_NUMBER, IMAGE_INDEX,
                            RELATIVE_PATH, IMAGE_STAGE_PATH, IMAGE_HASH, IMAGE_FORMAT,
                            WIDTH, HEIGHT
                        ) VALUES (
                            '{file_name_escaped}',
                            '{stage_path_escaped}',
                            {img['page_number']},
                            {img['image_index']},
                            '{relative_path_escaped}',
                            '{full_stage_path_escaped}',
                            '{img_hash}',
                            '{img_format}',
                            {width},
                            {height}
                        )
                        """
                        await self._run_in_thread(client.execute_statement, insert_sql)

                        async with upload_lock:
                            uploaded_count += 1

                        logger.debug(
                            "Uploaded image: page %d, image %d -> %s",
                            img["page_number"],
                            img["image_index"],
                            relative_path,
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to upload image page %d, image %d: %s",
                            img["page_number"],
                            img["image_index"],
                            e,
                        )

            # Run all uploads concurrently
            await asyncio.gather(*[upload_single_image(img) for img in images])

            logger.info(
                "Uploaded %d/%d images for %s", uploaded_count, len(images), file_name
            )
            return uploaded_count

        except Exception as e:
            logger.error("Failed to upload images: %s", e, exc_info=True)
            return 0

    def _chunk_document_content(
        self,
        pages: list[dict],
        chunk_size: int = 1000,
        overlap: int = 300,
    ) -> list[dict]:
        """Chunk document pages into overlapping text segments.

        Parameters
        ----------
        pages : list[dict]
            List of page dictionaries with 'page' and 'content' keys
        chunk_size : int, optional
            Target chunk size in characters, by default 2000
        overlap : int, optional
            Overlap between chunks in characters, by default 200

        Returns
        -------
        list[dict]
            List of chunk dictionaries with keys:
            - chunk_index: Sequential chunk number
            - chunk_text: Text content of chunk
            - start_char: Start position in page content
            - end_char: End position in page content
            - page_number: Source page number
        """
        chunks = []
        chunk_idx = 0

        for page_data in pages:
            page_num = page_data.get("page")
            content = page_data.get("content", "")

            if not content:
                continue

            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk_text = content[start:end]

                # Don't create tiny final chunks
                if len(chunk_text) < 100 and chunk_idx > 0:
                    # Merge with previous chunk if too small
                    if chunks:
                        chunks[-1]["chunk_text"] += " " + chunk_text
                        chunks[-1]["end_char"] = end
                    break

                chunks.append(
                    {
                        "chunk_index": chunk_idx,
                        "chunk_text": chunk_text,
                        "start_char": start,
                        "end_char": end,
                        "page_number": page_num,
                    }
                )

                chunk_idx += 1
                start = end - overlap

        logger.info(
            "Chunking complete: generated %d text chunks from %d pages",
            len(chunks),
            len(pages),
        )
        return chunks

    def _extract_images_from_pdf(
        self, pdf_bytes: bytes, min_size_kb: int = 2
    ) -> list[dict]:
        """Extract images from PDF - both embedded raster images and rendered figures.

        This method extracts:
        1. Embedded raster images (photos, logos) via pdfminer.six LTImage
        2. Rendered figures/charts from pages that have LTFigure elements but no raster images

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF file bytes
        min_size_kb : int, optional
            Minimum image size in KB to extract, by default 2

        Returns
        -------
        list[dict]
            List of image dictionaries with keys:
            - page_number: Source page number
            - image_index: Image index on page
            - image_bytes: Raw image bytes
            - image_hash: SHA256 hash for deduplication
            - width: Image width in pixels
            - height: Image height in pixels
            - format: Image format (png, jpeg, etc.)
        """
        from PIL import Image
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTImage, LTFigure

        images = []
        min_size_bytes = min_size_kb * 1024
        seen_hashes = set()  # For deduplication across pages
        pages_with_figures_no_images = []  # Track pages needing rendering

        def extract_images_from_element(element, page_num: int, img_counter: list):
            """Recursively extract raster images from layout elements."""
            if isinstance(element, LTImage):
                try:
                    stream = element.stream
                    if stream is not None:
                        img_data = stream.get_data()
                        if img_data and len(img_data) >= min_size_bytes:
                            img_hash = hashlib.sha256(img_data).hexdigest()
                            if img_hash in seen_hashes:
                                return
                            seen_hashes.add(img_hash)

                            try:
                                pil_img = Image.open(io.BytesIO(img_data))
                                width, height = pil_img.size
                                if width < 50 or height < 50:
                                    return

                                if pil_img.mode in ("RGBA", "P", "LA"):
                                    rgb_img = Image.new(
                                        "RGB", pil_img.size, (255, 255, 255)
                                    )
                                    if pil_img.mode == "P":
                                        pil_img = pil_img.convert("RGBA")
                                    rgb_img.paste(
                                        pil_img,
                                        mask=(
                                            pil_img.split()[-1]
                                            if pil_img.mode == "RGBA"
                                            else None
                                        ),
                                    )
                                    pil_img = rgb_img

                                img_buffer = io.BytesIO()
                                if len(img_data) > 1024 * 1024:
                                    pil_img.save(img_buffer, format="JPEG", quality=85)
                                    img_format = "jpeg"
                                else:
                                    pil_img.save(img_buffer, format="PNG")
                                    img_format = "png"
                                img_bytes = img_buffer.getvalue()

                                if len(img_bytes) > 5 * 1024 * 1024:
                                    return

                            except Exception as e:
                                logger.debug(
                                    "PIL failed to decode image on page %d: %s",
                                    page_num,
                                    e,
                                )
                                return

                            img_idx = img_counter[0]
                            img_counter[0] += 1
                            images.append(
                                {
                                    "page_number": page_num,
                                    "image_index": img_idx,
                                    "image_bytes": img_bytes,
                                    "image_hash": img_hash,
                                    "width": width,
                                    "height": height,
                                    "format": img_format,
                                }
                            )
                            logger.debug(
                                "Extracted raster image %d from page %d: %dx%d %s (%d KB)",
                                img_idx,
                                page_num,
                                width,
                                height,
                                img_format,
                                len(img_bytes) // 1024,
                            )
                except Exception as e:
                    logger.debug(
                        "Failed to extract image from page %d: %s", page_num, e
                    )

            elif isinstance(element, LTFigure):
                for child in element:
                    extract_images_from_element(child, page_num, img_counter)

            elif hasattr(element, "__iter__") and not isinstance(element, (str, bytes)):
                try:
                    for child in element:
                        extract_images_from_element(child, page_num, img_counter)
                except TypeError:
                    pass

        # Phase 1: Extract embedded raster images and identify pages with figures
        try:
            pdf_stream = io.BytesIO(pdf_bytes)
            page_count = 0
            img_counter = [0]

            for page_num, page_layout in enumerate(extract_pages(pdf_stream), start=1):
                page_count += 1
                page_img_count_before = len(images)
                has_figures = False

                for element in page_layout:
                    if isinstance(element, LTFigure):
                        has_figures = True
                    extract_images_from_element(element, page_num, img_counter)

                page_img_count = len(images) - page_img_count_before

                # Track pages that have figures but no extracted raster images
                if has_figures and page_img_count == 0:
                    pages_with_figures_no_images.append(page_num)

                if page_img_count > 0:
                    logger.debug(
                        "Page %d: extracted %d raster images", page_num, page_img_count
                    )

            logger.info(
                "Phase 1: Extracted %d raster images from PDF (%d pages)",
                len(images),
                page_count,
            )

        except Exception as e:
            logger.error(
                "Failed to extract raster images from PDF: %s", e, exc_info=True
            )

        # Phase 2: Render pages with figures but no raster images
        if pages_with_figures_no_images:
            logger.info(
                "Phase 2: Rendering %d pages with figures/charts: %s",
                len(pages_with_figures_no_images),
                pages_with_figures_no_images,
            )
            try:
                rendered = self._render_pdf_pages_as_images(
                    pdf_bytes,
                    pages_with_figures_no_images,
                    seen_hashes,
                    img_counter,
                )
                images.extend(rendered)
                logger.info("Phase 2: Rendered %d page images", len(rendered))
            except Exception as e:
                logger.error("Failed to render PDF pages: %s", e, exc_info=True)

        logger.info(
            "Total: Extracted %d images from PDF (%d raster + %d rendered)",
            len(images),
            len(images) - len([i for i in images if i.get("rendered")]),
            len([i for i in images if i.get("rendered")]),
        )
        return images

    def _render_pdf_pages_as_images(
        self,
        pdf_bytes: bytes,
        page_numbers: list[int],
        seen_hashes: set,
        img_counter: list,
        resolution: int = 150,
    ) -> list[dict]:
        """Render specific PDF pages as images using pdfplumber.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF bytes
        page_numbers : list[int]
            1-indexed page numbers to render
        seen_hashes : set
            Set of image hashes for deduplication
        img_counter : list
            Mutable counter for image indexing
        resolution : int
            DPI for rendering, by default 150

        Returns
        -------
        list[dict]
            List of rendered image dictionaries
        """
        from PIL import Image

        rendered_images = []

        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num in page_numbers:
                    try:
                        if page_num < 1 or page_num > len(pdf.pages):
                            continue

                        page = pdf.pages[page_num - 1]  # 0-indexed

                        # Render the page to an image
                        page_image = page.to_image(resolution=resolution)
                        pil_image = page_image.original

                        # Convert to RGB if needed
                        if pil_image.mode != "RGB":
                            pil_image = pil_image.convert("RGB")

                        # Save as JPEG (smaller than PNG for full pages)
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format="JPEG", quality=85)
                        img_bytes = img_buffer.getvalue()

                        # Skip if too small
                        if len(img_bytes) < 2048:  # 2KB min
                            continue

                        # Generate hash for deduplication
                        img_hash = hashlib.sha256(img_bytes).hexdigest()
                        if img_hash in seen_hashes:
                            continue
                        seen_hashes.add(img_hash)

                        width, height = pil_image.size
                        img_idx = img_counter[0]
                        img_counter[0] += 1

                        rendered_images.append(
                            {
                                "page_number": page_num,
                                "image_index": img_idx,
                                "image_bytes": img_bytes,
                                "image_hash": img_hash,
                                "width": width,
                                "height": height,
                                "format": "jpeg",
                                "rendered": True,  # Flag to indicate this is a rendered page
                            }
                        )
                        logger.debug(
                            "Rendered page %d as image %d: %dx%d jpeg (%d KB)",
                            page_num,
                            img_idx,
                            width,
                            height,
                            len(img_bytes) // 1024,
                        )

                    except Exception as e:
                        logger.debug("Failed to render page %d: %s", page_num, e)
                        continue

        except Exception as e:
            logger.error("Failed to open PDF for rendering: %s", e, exc_info=True)

        return rendered_images

    async def _generate_embeddings_for_document(
        self,
        client,
        file_name: str,
        stage_path: str,
        pages: list[dict],
        pdf_bytes: bytes | None = None,
        embed_images: bool = False,
        uploaded_images: list[dict] | None = None,
    ) -> bool:
        """Generate and store embeddings for document chunks and optionally images.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_name : str
            The document filename
        stage_path : str
            The stage path where document is stored
        pages : list[dict]
            List of parsed page dictionaries
        pdf_bytes : bytes | None, optional
            Raw PDF bytes for image extraction (only used if uploaded_images not provided), by default None
        embed_images : bool, optional
            Whether to extract and embed images, by default False
        uploaded_images : list[dict] | None, optional
            Pre-uploaded images with stage_path keys, avoids re-extracting, by default None

        Returns
        -------
        bool
            True if embeddings generated successfully
        """
        try:
            logger.info(
                "_generate_embeddings_for_document called: file=%s, embed_images=%s, has_pdf_bytes=%s, pdf_len=%d",
                file_name,
                embed_images,
                pdf_bytes is not None,
                len(pdf_bytes) if pdf_bytes else 0,
            )
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS"

            # Initialize counters
            text_embeddings_created = 0
            image_embeddings_created = 0

            # Ensure table exists
            await self._create_document_embeddings_table(client)

            # Check if embeddings already exist for this document
            file_name_escaped = file_name.replace("'", "''")
            check_sql = f"""
            SELECT COUNT(*) as cnt
            FROM {qualified_table}
            WHERE FILE_NAME = '{file_name_escaped}'
            """
            result = await self._run_in_thread(client.execute_statement, check_sql)
            if result:
                import json

                rows = json.loads(result) if isinstance(result, str) else []
                existing_count = rows[0].get("CNT", 0) if rows else 0
                if existing_count > 0:
                    logger.info(
                        "Found %d existing embeddings for %s, deleting...",
                        existing_count,
                        file_name,
                    )
                    delete_sql = f"""
                    DELETE FROM {qualified_table}
                    WHERE FILE_NAME = '{file_name_escaped}'
                    """
                    await self._run_in_thread(client.execute_statement, delete_sql)
                    logger.info(
                        "Deleted %d old embeddings for %s",
                        existing_count,
                        file_name,
                    )

            # Generate text chunks
            text_chunks = self._chunk_document_content(pages)
            logger.info(
                "Generated %d text chunks for document %s",
                len(text_chunks),
                file_name,
            )

            # Generate text embeddings in batches
            batch_size = 50
            for i in range(0, len(text_chunks), batch_size):
                batch = text_chunks[i : i + batch_size]
                for chunk in batch:
                    try:
                        chunk_text_escaped = chunk["chunk_text"].replace("'", "''")
                        file_name_escaped = file_name.replace("'", "''")
                        stage_path_escaped = stage_path.replace("'", "''")

                        embedding_id = hashlib.sha256(
                            f"{file_name}_{chunk['page_number']}_{chunk['chunk_index']}".encode()
                        ).hexdigest()

                        insert_sql = f"""
                        INSERT INTO {qualified_table}
                        (EMBEDDING_ID, FILE_NAME, STAGE_PATH, PAGE_NUMBER, CHUNK_INDEX,
                         CONTENT_TYPE, CHUNK_TEXT, CHUNK_START_CHAR, CHUNK_END_CHAR, EMBEDDING)
                        SELECT
                            '{embedding_id}',
                            '{file_name_escaped}',
                            '{stage_path_escaped}',
                            {chunk['page_number']},
                            {chunk['chunk_index']},
                            'text',
                            '{chunk_text_escaped}',
                            {chunk['start_char']},
                            {chunk['end_char']},
                            AI_EMBED(
                                'snowflake-arctic-embed-l-v2.0',
                                '{chunk_text_escaped}'
                            )
                        """
                        await self._run_in_thread(client.execute_statement, insert_sql)
                        text_embeddings_created += 1
                    except Exception as e:
                        logger.error(
                            "Failed to create text embedding for chunk %d: %s",
                            chunk["chunk_index"],
                            e,
                        )
                        continue

                logger.info(
                    "Text embeddings batch progress: %d/%d chunks processed",
                    min(i + batch_size, len(text_chunks)),
                    len(text_chunks),
                )

            logger.info(
                "Created %d text embeddings for document %s",
                text_embeddings_created,
                file_name,
            )

            # Generate image embeddings if enabled
            logger.info(
                "Image embedding check: embed_images=%s, pdf_bytes=%s, pdf_bytes_len=%d",
                embed_images,
                pdf_bytes is not None,
                len(pdf_bytes) if pdf_bytes else 0,
            )
            if embed_images and pdf_bytes:
                logger.info("Starting image extraction for %s", file_name)
                await self._create_document_images_stage(client)

                # Simple structure: @STAGE/filename/page_X_image_Y.format
                stage_name = "DOCUMENT_IMAGES"
                qualified_stage = f"OPENBB_AGENTS.{user_schema}.{stage_name}"

                # Clean up old images for this document (delete the folder)
                try:
                    file_name_escaped_path = file_name.replace("'", "''")
                    remove_sql = f"REMOVE @{qualified_stage}/{file_name_escaped_path}/"
                    await self._run_in_thread(client.execute_statement, remove_sql)
                    logger.info("Cleaned up old images for %s", file_name)
                except Exception as e:
                    logger.debug("Stage cleanup (may be empty): %s", e)

                images = self._extract_images_from_pdf(pdf_bytes)
                logger.info(
                    "Extracted %d images from document %s for embedding",
                    len(images),
                    file_name,
                )

                # Build a mapping of page number to page text for context
                page_text_map = {}
                for page_data in pages:
                    page_num = page_data.get("page")
                    content = page_data.get("content", "")
                    if page_num and content:
                        # Truncate to first 500 chars for context
                        page_text_map[page_num] = content[:500]

                image_batch_size = 10

                for i in range(0, len(images), image_batch_size):
                    batch = images[i : i + image_batch_size]
                    for img in batch:
                        try:
                            # Simple path: filename/page_X_image_Y.format
                            img_filename = f"{file_name}/page_{img['page_number']}_image_{img['image_index']}.{img['format']}"

                            # Upload image bytes to stage
                            uploaded_path = await self._run_in_thread(
                                client.upload_bytes_to_stage,
                                img["image_bytes"],
                                img_filename,
                                stage_name,
                            )
                            logger.info(
                                "Uploaded image %s to stage path: %s",
                                img_filename,
                                uploaded_path,
                            )

                            # Create embedding from staged image
                            embedding_id = hashlib.sha256(
                                f"{file_name}_{img['page_number']}_img_{img['image_index']}".encode()
                            ).hexdigest()

                            file_name_escaped = file_name.replace("'", "''")
                            stage_path_escaped = stage_path.replace("'", "''")
                            img_filename_escaped = img_filename.replace("'", "''")

                            # Get page text for context (link image to text)
                            page_text = page_text_map.get(img["page_number"], "")
                            page_text_escaped = page_text.replace("'", "''")

                            # Store in DOCUMENT_EMBEDDINGS with all links:
                            # - FILE_NAME: the document
                            # - PAGE_NUMBER: which page
                            # - IMAGE_STAGE_PATH: where the image file is
                            insert_sql = f"""
                            INSERT INTO {qualified_table}
                            (EMBEDDING_ID, FILE_NAME, STAGE_PATH, PAGE_NUMBER, CHUNK_INDEX,
                             CONTENT_TYPE, CHUNK_TEXT, IMAGE_STAGE_PATH, IMAGE_HASH, EMBEDDING, EMBEDDING_MODEL)
                            SELECT
                                '{embedding_id}',
                                '{file_name_escaped}',
                                '{stage_path_escaped}',
                                {img['page_number']},
                                {img['image_index']},
                                'image',
                                '{page_text_escaped}',
                                '@{qualified_stage}/{img_filename_escaped}',
                                '{img['image_hash']}',
                                AI_EMBED(
                                    'voyage-multimodal-3',
                                    TO_FILE('@{qualified_stage}', '{img_filename_escaped}')
                                ),
                                'voyage-multimodal-3'
                            """
                            await self._run_in_thread(
                                client.execute_statement, insert_sql
                            )
                            image_embeddings_created += 1
                        except Exception as e:
                            logger.error(
                                "Failed to create image embedding for page %d, image %d: %s",
                                img["page_number"],
                                img["image_index"],
                                e,
                            )
                            continue

                    logger.info(
                        "Image embeddings batch progress: %d/%d images processed",
                        min(i + image_batch_size, len(images)),
                        len(images),
                    )

                logger.info(
                    "Created %d image embeddings for document %s",
                    image_embeddings_created,
                    file_name,
                )

            logger.info(
                "Embedding generation complete for document %s (text=%d, images=%d)",
                file_name,
                text_embeddings_created,
                image_embeddings_created if embed_images else 0,
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to generate embeddings for document %s: %s",
                file_name,
                e,
                exc_info=True,
            )
            return False

    async def load_pdf_positions_from_snowflake(
        self,
        client,
        conversation_id: str,
        file_name: str | None = None,
        stage_path: str | None = None,
    ) -> bool:
        """Load PDF text positions from Snowflake into cache.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str | None, optional
            Optional filename to filter by
        stage_path : str | None, optional
            Optional stage path to filter by

        Returns
        -------
        bool
            True if positions loaded successfully, False otherwise
        """
        if not file_name and not stage_path:
            return False

        user_schema = await self._get_user_schema(client)
        where_clause = []
        if file_name:
            escaped_file = file_name.replace("'", "''")
            where_clause.append(f"FILE_NAME = '{escaped_file}'")
        if stage_path:
            escaped_stage = stage_path.replace("'", "''")
            where_clause.append(f"STAGE_PATH = '{escaped_stage}'")

        query = f"""
        SELECT PAGE, TEXT, X0, TOP, X1, BOTTOM
        FROM OPENBB_AGENTS.{user_schema}.PDF_TEXT_POSITIONS
        WHERE {' OR '.join(where_clause)}
        ORDER BY PAGE, TOP
        """

        result = await self._run_in_thread(client.execute_query, query)
        result_json = json.loads(result)
        row_data = result_json.get("rowData", [])
        if not row_data:

            return False

        positions = []
        for row in row_data:
            # Handle both uppercase and lowercase keys
            page = row.get("PAGE") if "PAGE" in row else row.get("page")
            text = row.get("TEXT") if "TEXT" in row else row.get("text", "")
            x0 = row.get("X0") if "X0" in row else row.get("x0")
            top = row.get("TOP") if "TOP" in row else row.get("top")
            x1 = row.get("X1") if "X1" in row else row.get("x1")
            bottom = row.get("BOTTOM") if "BOTTOM" in row else row.get("bottom")

            positions.append(
                {
                    "page": page,
                    "text": str(text) if text is not None else "",
                    "x0": float(x0) if x0 is not None else None,
                    "top": float(top) if top is not None else None,
                    "x1": float(x1) if x1 is not None else None,
                    "bottom": float(bottom) if bottom is not None else None,
                }
            )

        self.set_pdf_blocks(conversation_id, positions)
        return True

    async def _extract_positions_from_stage(
        self,
        client,
        conversation_id: str,
        file_name: str,
        stage_path: str,
    ) -> bool:
        """Extract PDF positions from stage and store them.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str
            The PDF filename
        stage_path : str
            The stage path

        Returns
        -------
        bool
            True if extraction and storage successful, False otherwise
        """
        try:
            # Check document source cache first
            doc_source = self.get_document_source(conversation_id) or {}
            pdf_bytes = doc_source.get("widget_pdf_bytes")
            if pdf_bytes:
                logger.debug(
                    "Using cached PDF bytes in _extract_positions_from_stage (%d bytes)",
                    len(pdf_bytes),
                )
            else:
                logger.debug(
                    "Downloading PDF %s in _extract_positions_from_stage (no cache)",
                    file_name,
                )
                pdf_base64 = await self._run_in_thread(
                    client.download_cortex_document,
                    file_name,
                )
                pdf_bytes = base64.b64decode(pdf_base64)
                # Cache for future use
                doc_source["widget_pdf_bytes"] = pdf_bytes
                self.set_document_source(conversation_id, doc_source)

            _, text_positions, _ = self.extract_pdf_with_positions(pdf_bytes)
            await self.store_pdf_positions_in_snowflake(
                client,
                file_name,
                stage_path,
                text_positions,
            )
            self.set_pdf_blocks(conversation_id, text_positions)
            return True
        except Exception as exc:  # noqa: W0718
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Failed to extract PDF positions: %s", exc)
            return False

    async def ensure_pdf_positions(
        self,
        client,
        conversation_id: str,
        file_name: str,
        stage_path: str | None = None,
    ) -> bool:
        """Ensure PDF positions are loaded and cached for a conversation.

        Tries to load from cache, then from Snowflake, then extracts from stage.
        Uses locking to prevent duplicate work.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str
            The PDF filename
        stage_path : str | None, optional
            Optional stage path for extraction

        Returns
        -------
        bool
            True if positions are available, False otherwise
        """
        if not file_name:
            return False

        if self.get_pdf_blocks(conversation_id):
            return True

        key = f"pdf::{conversation_id}::{file_name}"
        lock = self._get_lock(key)
        async with lock:
            if self.get_pdf_blocks(conversation_id):
                return True
            loaded = await self.load_pdf_positions_from_snowflake(
                client, conversation_id, file_name=file_name, stage_path=stage_path
            )
            if loaded:
                return True
            if stage_path and file_name.lower().endswith(".pdf"):
                return await self._extract_positions_from_stage(
                    client, conversation_id, file_name, stage_path
                )

            return False

    # ------------------------------------------------------------------
    # Snowflake document pages
    # ------------------------------------------------------------------
    async def load_snowflake_document_pages(
        self,
        client,
        conversation_id: str,
        file_name: str | None = None,
    ) -> bool:
        """Load parsed document pages from Snowflake DOCUMENT_PARSE_RESULTS.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str | None, optional
            Optional filename to filter by (otherwise loads latest)

        Returns
        -------
        bool
            True if pages loaded successfully, False otherwise
        """
        user_schema = await self._get_user_schema(client)
        qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS"

        if file_name:
            escaped_filename = file_name.replace("'", "''")
            query = f"""
            SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
            FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}'
            ORDER BY PAGE_NUMBER
            """
        else:
            query = f"""
            SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
            FROM {qualified_table}
            WHERE PARSED_AT = (SELECT MAX(PARSED_AT) FROM {qualified_table})
            ORDER BY PAGE_NUMBER
            """

        result_json = await self._run_in_thread(client.execute_statement, query)
        rows = json.loads(result_json) if result_json else []
        if not rows:
            return False

        pages = []
        detected_file_name = None
        for row in rows:
            fn = row.get("FILE_NAME") or row.get("file_name")
            page_num = row.get("PAGE_NUMBER") or row.get("page_number")
            content = row.get("PAGE_CONTENT") or row.get("page_content")
            if fn and not detected_file_name:
                detected_file_name = fn
            if page_num and content:
                pages.append(
                    {
                        "page": int(page_num),
                        "content": content,
                        "file_name": fn or file_name or "unknown",
                    }
                )

        if pages:
            self.set_document_pages(conversation_id, pages)
            return True
        return False

    async def ensure_document_pages(
        self,
        client,
        conversation_id: str,
        file_name: str,
    ) -> bool:
        """Ensure document pages are loaded and cached for a conversation.

        Uses locking to prevent duplicate work.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str
            The document filename

        Returns
        -------
        bool
            True if pages are available, False otherwise
        """
        if self.get_document_pages(conversation_id):
            return True
        key = f"pages::{conversation_id}::{file_name}"
        lock = self._get_lock(key)
        async with lock:
            if self.get_document_pages(conversation_id):
                return True
            return await self.load_snowflake_document_pages(
                client, conversation_id, file_name
            )

    # ------------------------------------------------------------------
    # Document metadata storage (TOC/outline)
    # ------------------------------------------------------------------
    async def store_document_metadata(
        self,
        client,
        file_name: str,
        stage_path: str,
        metadata: dict,
    ) -> bool:
        """Store PDF metadata (outline, info) in Snowflake DOCUMENT_PARSE_RESULTS.

        Stores metadata as PAGE_NUMBER=0 row with JSON in METADATA column.
        This avoids needing to re-extract metadata from PDF on each load.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_name : str
            The document filename
        stage_path : str
            The stage path where document is stored
        metadata : dict
            Metadata dict from extract_pdf_metadata()

        Returns
        -------
        bool
            True if storage successful, False otherwise
        """
        if not metadata:
            return False

        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS"

            # Delete any existing metadata row for this file
            escaped_filename = file_name.replace("'", "''")
            delete_sql = f"""
            DELETE FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}' AND PAGE_NUMBER = 0
            """
            await self._run_in_thread(client.execute_statement, delete_sql)

            # Prepare metadata JSON - only store outline and info (not page_summaries to save space)
            metadata_to_store = {
                "page_count": metadata.get("page_count", 0),
                "outline": metadata.get("outline", []),
                "info": metadata.get("info", {}),
            }
            metadata_json = json.dumps(metadata_to_store).replace("'", "''")
            escaped_stage = stage_path.replace("'", "''")

            # Insert metadata row with PAGE_NUMBER=0
            # Use SELECT with PARSE_JSON instead of VALUES clause (Snowflake limitation)
            insert_sql = f"""
            INSERT INTO {qualified_table}
            (FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, METADATA)
            SELECT
                '{escaped_filename}',
                '{escaped_stage}',
                0,
                '',
                PARSE_JSON('{metadata_json}')
            """
            await self._run_in_thread(client.execute_statement, insert_sql)

            logger.debug(
                "Stored metadata for %s: %d outline entries",
                file_name,
                len(metadata_to_store.get("outline", [])),
            )
            return True

        except Exception as e:
            logger.warning("Failed to store document metadata: %s", e)
            return False

    async def load_document_metadata_from_snowflake(
        self,
        client,
        conversation_id: str,
        file_name: str,
    ) -> bool:
        """Load stored PDF metadata from Snowflake into cache.

        Queries DOCUMENT_PARSE_RESULTS for PAGE_NUMBER=0 row containing
        pre-extracted outline and info. Avoids re-downloading PDF just for metadata.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        file_name : str
            The document filename

        Returns
        -------
        bool
            True if metadata loaded from Snowflake, False otherwise
        """
        if not file_name:
            return False

        # Check if already cached
        if self.get_pdf_metadata(conversation_id):
            return True

        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS"
            escaped_filename = file_name.replace("'", "''")

            query = f"""
            SELECT METADATA
            FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}' AND PAGE_NUMBER = 0
            LIMIT 1
            """

            result_json = await self._run_in_thread(client.execute_statement, query)
            rows = json.loads(result_json) if result_json else []

            if not rows:
                return False

            # Parse metadata from VARIANT column
            row = rows[0]
            metadata_raw = row.get("METADATA") or row.get("metadata")

            if not metadata_raw:
                return False

            # Handle if it's already a dict or needs JSON parsing
            if isinstance(metadata_raw, str):
                stored_metadata = json.loads(metadata_raw)
            else:
                stored_metadata = metadata_raw

            # Reconstruct full metadata structure
            # Page summaries are not stored (too large), regenerate if needed later
            metadata = {
                "page_count": stored_metadata.get("page_count", 0),
                "outline": stored_metadata.get("outline", []),
                "page_summaries": [],  # Not stored, will be empty
                "info": stored_metadata.get("info", {}),
            }

            self.set_pdf_metadata(conversation_id, metadata)
            logger.debug(
                "Loaded metadata from Snowflake for %s: %d outline entries",
                file_name,
                len(metadata.get("outline", [])),
            )
            return True

        except Exception as e:
            logger.debug("Could not load metadata from Snowflake: %s", e)
            return False

    # ------------------------------------------------------------------
    # Citation helpers
    # ------------------------------------------------------------------
    def find_best_match(
        self,
        query_text: str,
        pdf_positions: list[dict],
        preferred_page: int | None = None,
        require_exact_phrases: bool = True,
    ) -> dict | None:
        """Find the best matching text position for a query in PDF positions.

        Uses fuzzy matching with stopword filtering, similarity scoring, and exact phrase matching.

        Parameters
        ----------
        query_text : str
            The text to search for
        pdf_positions : list[dict]
            List of position dictionaries to search in
        preferred_page : int | None, optional
            Optional page number to boost in scoring
        require_exact_phrases : bool, optional
            Whether to boost scores for exact phrase matches, by default True

        Returns
        -------
        dict | None
            Best matching position dictionary or None if no good match found
        """
        if not query_text or not pdf_positions:
            return None

        query_lower = query_text.lower().strip()

        # Extract key phrases (3+ consecutive words)
        key_phrases = re.findall(r"\b\w+(?:\s+\w+){2,}\b", query_lower)

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
        query_words = [
            word
            for word in re.findall(r"\b[a-z]+\b", query_lower)
            if word not in stopwords and len(word) > 2
        ]
        if not query_words:
            return None

        best_match = None
        best_score = 0
        best_on_preferred_page = None
        best_score_on_preferred = 0

        for position in pdf_positions:
            line_lower = position["text"].lower()
            is_preferred_page = (
                preferred_page and position.get("page") == preferred_page
            )

            # Check for exact phrase matches (STRONG signal)
            exact_phrase_bonus = 0
            if require_exact_phrases and key_phrases:
                for phrase in key_phrases[:3]:  # Check top 3 key phrases
                    if phrase in line_lower:
                        exact_phrase_bonus += 50

            matching_words = sum(1 for word in query_words if word in line_lower)
            similarity_ratio = SequenceMatcher(None, query_lower, line_lower).ratio()
            score = (
                (matching_words * 10) + (similarity_ratio * 100) + exact_phrase_bonus
            )

            # Track best match on preferred page separately
            if is_preferred_page:
                if score > best_score_on_preferred:
                    best_score_on_preferred = score
                    best_on_preferred_page = position

            # Global best match (with boost for preferred page)
            if is_preferred_page:
                score *= 2.0  # Significantly boost preferred page matches
            if score > best_score:
                best_score = score
                best_match = position

        # If we have a preferred page, return match or page-only reference (NO fake rectangles)
        if preferred_page:
            # If we found a good match on the preferred page, use it
            if best_on_preferred_page and best_score_on_preferred >= 30:
                logger.debug(
                    "Found match on preferred page %d with score %d",
                    preferred_page,
                    best_score_on_preferred,
                )
                return best_on_preferred_page

            # Return page reference WITHOUT coordinates - no fake rectangles!
            # The streaming handler will create a page-only citation without highlight
            logger.debug(
                "No specific match on page %d, returning page-only reference (no highlight)",
                preferred_page,
            )
            return {
                "page": preferred_page,
                "text": "",
                # NO x0, top, x1, bottom - signals no highlight should be drawn
            }

        # No preferred page: require high score to avoid false positives
        min_score = 60 if require_exact_phrases else 30
        if best_score < min_score:
            logger.debug(
                "No good match found for citation. Best score: %d (min: %d)",
                best_score,
                min_score,
            )
            return None
        return best_match

    def find_all_related_spans(
        self,
        query_text: str,
        pdf_positions: list[dict],
        anchor_position: dict,
    ) -> list[dict]:
        """Find all related text spans near an anchor position.

        Finds text positions on the same or adjacent pages that match the query.

        Parameters
        ----------
        query_text : str
            The text to search for
        pdf_positions : list[dict]
            List of position dictionaries
        anchor_position : dict
            The primary matching position

        Returns
        -------
        list[dict]
            List of up to 5 related position dictionaries, sorted by position
        """
        from difflib import SequenceMatcher

        if not query_text or not pdf_positions or not anchor_position:
            return []

        query_lower = query_text.lower().strip()
        anchor_page = anchor_position.get("page")
        if anchor_page is None:
            return []
        anchor_text = anchor_position.get("text", "")
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
        query_words = [
            word
            for word in re.findall(r"\b[a-z]+\b", query_lower)
            if word not in stopwords and len(word) > 2
        ]
        if not query_words:
            return []

        related_spans = []
        for position in pdf_positions:
            if (
                position.get("text") == anchor_text
                and position.get("page") == anchor_page
            ):
                continue
            pos_page = position.get("page")
            if pos_page is None:
                continue
            if abs(pos_page - anchor_page) > 1:
                continue
            line_lower = position["text"].lower()
            matching_words = sum(1 for word in query_words if word in line_lower)
            similarity_ratio = SequenceMatcher(None, query_lower, line_lower).ratio()
            score = (matching_words * 10) + (similarity_ratio * 100)
            if pos_page == anchor_page:
                score *= 1.2
            if score >= 20:
                position["related_score"] = score
                related_spans.append(position)

        related_spans.sort(
            key=lambda x: (x.get("page", 0), x.get("top", 0), x.get("x0", 0))
        )
        return related_spans[:5]

    def find_best_match_in_snowflake_pages(
        self,
        query_text: str,
        conversation_id: str,
        preferred_page: int | None = None,
    ) -> dict | None:
        """Find best matching page in Snowflake-parsed document pages.

        Uses EXACT substring matching first, then falls back to fuzzy matching.

        Parameters
        ----------
        query_text : str
            The text to search for
        conversation_id : str
            The conversation identifier
        preferred_page : int | None, optional
            Optional page number to boost in scoring

        Returns
        -------
        dict | None
            Dictionary with match info or None if no good match found
        """
        from difflib import SequenceMatcher

        pages = self.get_document_pages(conversation_id)
        if not pages or not query_text:
            return None

        query_lower = query_text.lower().strip()

        # STEP 1: Try EXACT substring match first (most reliable)
        # Extract meaningful phrases (5+ words) to match exactly
        words = query_lower.split()
        exact_match_page = None

        # Try progressively shorter phrases
        for phrase_len in [8, 6, 5, 4]:
            if len(words) >= phrase_len:
                for i in range(len(words) - phrase_len + 1):
                    phrase = " ".join(words[i : i + phrase_len])
                    # Skip if phrase is too generic
                    if len(phrase) < 20:
                        continue
                    for page_data in pages:
                        if phrase in page_data["content"].lower():
                            exact_match_page = page_data
                            break
                    if exact_match_page:
                        break
            if exact_match_page:
                break

        if exact_match_page:
            return {
                "text": (
                    exact_match_page["content"][:200] + "..."
                    if len(exact_match_page["content"]) > 200
                    else exact_match_page["content"]
                ),
                "page": exact_match_page["page"],
                "source": "snowflake_exact",
                "file_name": exact_match_page.get("file_name", ""),
            }

        # STEP 2: Fall back to fuzzy matching with stricter thresholds
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
        query_words = [
            word
            for word in re.findall(r"\b[a-z]+\b", query_lower)
            if word not in stopwords and len(word) > 2
        ]
        if not query_words:
            return None

        best_match = None
        best_score = 0
        for page_data in pages:
            content_lower = page_data["content"].lower()
            matching_words = sum(1 for word in query_words if word in content_lower)
            similarity_ratio = SequenceMatcher(
                None, query_lower, content_lower[:500]
            ).ratio()
            score = (matching_words * 10) + (similarity_ratio * 100)
            if preferred_page and page_data["page"] == preferred_page:
                score *= 1.5
            if score > best_score:
                best_score = score
                best_match = page_data

        if best_score < 30:
            return None
        if best_match:
            return {
                "text": (
                    best_match["content"][:200] + "..."
                    if len(best_match["content"]) > 200
                    else best_match["content"]
                ),
                "page": best_match["page"],
                "source": "snowflake",
                "file_name": best_match["file_name"],
            }
        return None

    async def semantic_search_documents(
        self,
        client,
        query: str,
        file_name: str | None = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_images: bool = True,
    ) -> tuple[list[dict], float]:
        """Search documents using semantic similarity via vector embeddings.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        query : str
            The search query text
        file_name : str | None, optional
            Optional filename to filter results, by default None
        top_k : int, optional
            Number of top results to return, by default 5
        similarity_threshold : float, optional
            Initial minimum similarity score (-1 to 1), by default 0.7.
            Will progressively lower to 0.5, then 0.4 if no results found.
        include_images : bool, optional
            Whether to include image embedding results, by default True

        Returns
        -------
        tuple[list[dict], float]
            Tuple of (list of matching chunks with similarity scores, final threshold used)
        """
        # Progressive thresholds to try
        thresholds_to_try = [similarity_threshold, 0.5, 0.4]
        # Remove duplicates and keep only thresholds <= initial
        thresholds_to_try = sorted(
            set(t for t in thresholds_to_try if t <= similarity_threshold), reverse=True
        )

        try:
            user_schema = await self._get_user_schema(client)
            qualified_table = f"OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS"

            # Build query embedding
            query_escaped = query.replace("'", "''")

            # Build file filter
            file_filter = ""
            if file_name:
                file_name_escaped = file_name.replace("'", "''")
                file_filter = f"AND FILE_NAME = '{file_name_escaped}'"

            for current_threshold in thresholds_to_try:
                # Search TEXT embeddings using snowflake-arctic-embed model
                text_search_sql = f"""
                SELECT * FROM (
                    SELECT
                        EMBEDDING_ID,
                        FILE_NAME,
                        STAGE_PATH,
                        PAGE_NUMBER,
                        CHUNK_INDEX,
                        CONTENT_TYPE,
                        CHUNK_TEXT,
                        IMAGE_STAGE_PATH,
                        CHUNK_START_CHAR,
                        CHUNK_END_CHAR,
                        VECTOR_COSINE_SIMILARITY(
                            EMBEDDING,
                            AI_EMBED('snowflake-arctic-embed-l-v2.0', '{query_escaped}')
                        ) AS SIMILARITY_SCORE
                    FROM {qualified_table}
                    WHERE CONTENT_TYPE = 'text' {file_filter}
                ) subq
                WHERE SIMILARITY_SCORE >= {current_threshold}
                ORDER BY SIMILARITY_SCORE DESC
                LIMIT {top_k}
                """

                result = await self._run_in_thread(
                    client.execute_statement, text_search_sql
                )

                # execute_statement returns JSON string, not a cursor
                all_matches = []

                if result:
                    try:
                        parsed_result = (
                            json.loads(result) if isinstance(result, str) else result
                        )
                        rows = []
                        if isinstance(parsed_result, dict):
                            rows = parsed_result.get(
                                "data", parsed_result.get("DATA", [])
                            )
                        elif isinstance(parsed_result, list):
                            rows = parsed_result

                        all_matches.extend(self._parse_embedding_rows(rows))
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse text search result as JSON")

                # Search IMAGE embeddings using voyage-multimodal-3 model
                if include_images:
                    image_search_sql = f"""
                    SELECT * FROM (
                        SELECT
                            EMBEDDING_ID,
                            FILE_NAME,
                            STAGE_PATH,
                            PAGE_NUMBER,
                            CHUNK_INDEX,
                            CONTENT_TYPE,
                            CHUNK_TEXT,
                            IMAGE_STAGE_PATH,
                            CHUNK_START_CHAR,
                            CHUNK_END_CHAR,
                            VECTOR_COSINE_SIMILARITY(
                                EMBEDDING,
                                AI_EMBED('voyage-multimodal-3', '{query_escaped}')
                            ) AS SIMILARITY_SCORE
                        FROM {qualified_table}
                        WHERE CONTENT_TYPE = 'image' {file_filter}
                    ) subq
                    WHERE SIMILARITY_SCORE >= {current_threshold}
                    ORDER BY SIMILARITY_SCORE DESC
                    LIMIT {top_k}
                    """

                    image_result = await self._run_in_thread(
                        client.execute_statement, image_search_sql
                    )

                    if image_result:
                        try:
                            parsed_image_result = (
                                json.loads(image_result)
                                if isinstance(image_result, str)
                                else image_result
                            )
                            image_rows = []
                            if isinstance(parsed_image_result, dict):
                                image_rows = parsed_image_result.get(
                                    "data", parsed_image_result.get("DATA", [])
                                )
                            elif isinstance(parsed_image_result, list):
                                image_rows = parsed_image_result

                            all_matches.extend(self._parse_embedding_rows(image_rows))
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse image search result as JSON"
                            )

                if not all_matches:
                    logger.info(
                        "No results at threshold %.2f for query '%s', trying lower...",
                        current_threshold,
                        query[:50],
                    )
                    continue

                # Sort combined results by similarity score and limit to top_k
                all_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
                matches = all_matches[:top_k]

                logger.info(
                    "Semantic search found %d results for query '%s' (threshold=%.2f, text=%d, images=%d)",
                    len(matches),
                    query[:50],
                    current_threshold,
                    len([m for m in matches if m.get("content_type") == "text"]),
                    len([m for m in matches if m.get("content_type") == "image"]),
                )
                return matches, current_threshold

            # No results at any threshold
            logger.warning(
                "No semantic search results at any threshold for query: %s", query[:50]
            )
            return [], (
                thresholds_to_try[-1] if thresholds_to_try else similarity_threshold
            )

        except Exception as e:
            error_str = str(e)
            # Check for table not found error (Snowflake error 002003)
            if "002003" in error_str or "does not exist" in error_str.lower():
                logger.warning(
                    "Document embeddings table does not exist yet: %s", error_str
                )
                return [], similarity_threshold
            logger.error("Semantic search failed: %s", e, exc_info=True)
            return [], similarity_threshold

    def _parse_embedding_rows(self, rows: list) -> list[dict]:
        """Parse embedding result rows into standardized dictionaries.

        Parameters
        ----------
        rows : list
            List of rows from embedding search query

        Returns
        -------
        list[dict]
            Parsed embedding result dictionaries
        """
        matches = []
        for row in rows:
            # Handle both list (positional) and dict (keyed) formats
            if isinstance(row, (list, tuple)):
                matches.append(
                    {
                        "embedding_id": row[0],
                        "file_name": row[1],
                        "stage_path": row[2],
                        "page_number": row[3],
                        "chunk_index": row[4],
                        "content_type": row[5],
                        "chunk_text": row[6],
                        "image_stage_path": row[7],
                        "chunk_start_char": row[8],
                        "chunk_end_char": row[9],
                        "similarity_score": float(row[10]) if len(row) > 10 else 0.0,
                    }
                )
            elif isinstance(row, dict):
                # Handle both uppercase and lowercase keys
                matches.append(
                    {
                        "embedding_id": row.get("EMBEDDING_ID")
                        or row.get("embedding_id"),
                        "file_name": row.get("FILE_NAME") or row.get("file_name"),
                        "stage_path": row.get("STAGE_PATH") or row.get("stage_path"),
                        "page_number": row.get("PAGE_NUMBER") or row.get("page_number"),
                        "chunk_index": row.get("CHUNK_INDEX") or row.get("chunk_index"),
                        "content_type": row.get("CONTENT_TYPE")
                        or row.get("content_type"),
                        "chunk_text": row.get("CHUNK_TEXT") or row.get("chunk_text"),
                        "image_stage_path": row.get("IMAGE_STAGE_PATH")
                        or row.get("image_stage_path"),
                        "chunk_start_char": row.get("CHUNK_START_CHAR")
                        or row.get("chunk_start_char"),
                        "chunk_end_char": row.get("CHUNK_END_CHAR")
                        or row.get("chunk_end_char"),
                        "similarity_score": float(
                            row.get("SIMILARITY_SCORE")
                            or row.get("similarity_score")
                            or 0
                        ),
                    }
                )
        return matches

    async def backfill_embeddings_for_existing_documents(
        self,
        client,
        file_names: list[str] | None = None,
        batch_size: int = 3,
        embed_images: bool = False,
    ) -> dict:
        """Backfill embeddings for existing documents that don't have them.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_names : list[str] | None, optional
            Specific files to backfill, or None for all missing, by default None
        batch_size : int, optional
            Number of documents to process in parallel, by default 3
        embed_images : bool, optional
            Whether to extract and embed images, by default False

        Returns
        -------
        dict
            Summary with keys: processed, succeeded, failed, skipped
        """
        try:
            user_schema = await self._get_user_schema(client)

            # Ensure table exists
            await self._create_document_embeddings_table(client)

            # Find documents without embeddings
            file_filter = ""
            if file_names:
                file_names_escaped = [f.replace("'", "''") for f in file_names]
                file_list = "','".join(file_names_escaped)
                file_filter = f"AND FILE_NAME IN ('{file_list}')"

            find_missing_sql = f"""
            SELECT DISTINCT FILE_NAME, STAGE_PATH
            FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS
            WHERE FILE_NAME NOT IN (
                SELECT DISTINCT FILE_NAME
                FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_EMBEDDINGS
            )
            {file_filter}
            """

            result = await self._run_in_thread(
                client.execute_statement, find_missing_sql
            )

            if not result or not hasattr(result, "fetchall"):
                logger.info("No documents found for embedding backfill")
                return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

            missing_docs = result.fetchall()
            logger.info(
                "Found %d documents without embeddings to backfill",
                len(missing_docs),
            )

            summary = {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

            # Process in batches
            for i in range(0, len(missing_docs), batch_size):
                batch = missing_docs[i : i + batch_size]

                for file_name, stage_path in batch:
                    summary["processed"] += 1
                    logger.info(
                        "Backfilling embeddings for document %s (%d/%d)",
                        file_name,
                        summary["processed"],
                        len(missing_docs),
                    )

                    try:
                        # Get parsed pages from results table
                        pages_sql = f"""
                        SELECT PAGE_NUMBER, PAGE_CONTENT
                        FROM OPENBB_AGENTS.{user_schema}.DOCUMENT_PARSE_RESULTS
                        WHERE FILE_NAME = '{file_name.replace("'", "''")}'
                        ORDER BY PAGE_NUMBER
                        """
                        pages_result = await self._run_in_thread(
                            client.execute_statement, pages_sql
                        )

                        if not pages_result or not hasattr(pages_result, "fetchall"):
                            logger.warning("No pages found for document %s", file_name)
                            summary["skipped"] += 1
                            continue

                        pages_rows = pages_result.fetchall()
                        pages = [
                            {"page": row[0], "content": row[1]} for row in pages_rows
                        ]

                        # Download PDF if images needed
                        pdf_bytes = None
                        if embed_images:
                            try:
                                download_sql = f"""
                                SELECT GET_PRESIGNED_URL('{stage_path}', 3600)
                                """
                                url_result = await self._run_in_thread(
                                    client.execute_statement, download_sql
                                )
                                if url_result and hasattr(url_result, "fetchone"):
                                    url = url_result.fetchone()[0]
                                    import httpx

                                    async with httpx.AsyncClient() as http_client:
                                        response = await http_client.get(url)
                                        pdf_bytes = response.content
                            except Exception as e:
                                logger.warning(
                                    "Failed to download PDF for image extraction: %s", e
                                )

                        # Generate embeddings
                        success = await self._generate_embeddings_for_document(
                            client=client,
                            file_name=file_name,
                            stage_path=stage_path,
                            pages=pages,
                            pdf_bytes=pdf_bytes,
                            embed_images=embed_images,
                        )

                        if success:
                            summary["succeeded"] += 1
                            logger.info(
                                "Successfully backfilled embeddings for %s", file_name
                            )
                        else:
                            summary["failed"] += 1
                            logger.warning(
                                "Failed to backfill embeddings for %s", file_name
                            )

                    except Exception as e:
                        summary["failed"] += 1
                        logger.error(
                            "Error backfilling embeddings for %s: %s",
                            file_name,
                            e,
                            exc_info=True,
                        )

                logger.info(
                    "Backfill batch progress: %d/%d documents processed",
                    min(i + batch_size, len(missing_docs)),
                    len(missing_docs),
                )

            logger.info(
                "Backfill complete: processed=%d, succeeded=%d, failed=%d, skipped=%d",
                summary["processed"],
                summary["succeeded"],
                summary["failed"],
                summary["skipped"],
            )
            return summary

        except Exception as e:
            logger.error("Backfill embeddings failed: %s", e, exc_info=True)
            return {"processed": 0, "succeeded": 0, "failed": 0, "skipped": 0}

    def find_quote_in_pdf_blocks(self, quote: str, conversation_id: str) -> dict | None:
        """Find a quote in cached PDF text blocks.

        Tries exact match, then prefix match, then keyword match.

        Parameters
        ----------
        quote : str
            The quote text to find
        conversation_id : str
            The conversation identifier

        Returns
        -------
        dict | None
            Position dictionary or None if not found
        """
        pdf_positions = self.get_pdf_blocks(conversation_id)
        if not pdf_positions:
            return None
        quote_lower = quote.lower().strip()
        for position in pdf_positions:
            if quote_lower in position["text"].lower():
                return position
        if len(quote) > 50:
            quote_start = quote_lower[:50]
            for position in pdf_positions:
                if quote_start in position["text"].lower():
                    return position
        quote_words = quote_lower.split()
        if len(quote_words) > 5:
            key_words = quote_words[:5]
            for position in pdf_positions:
                text_lower = position["text"].lower()
                if all(word in text_lower for word in key_words):
                    return position
        return None

    @staticmethod
    def extract_quotes_from_llm_response(response_text: str) -> list[tuple[str, int]]:
        """Extract quotes and citation numbers from LLM response text.

        Supports formats like "text[1]" and "quote" (Page 2).

        Parameters
        ----------
        response_text : str
            The LLM response text

        Returns
        -------
        list[tuple[str, int]]
            List of tuples (quote_text, citation_number)
        """
        citations = []
        pattern = r"([^.!?]*?)\[(\d+)\]"
        matches = re.findall(pattern, response_text)
        for context_text, citation_num in matches:
            sentence = context_text.strip()
            if len(sentence) > 200:
                sentence = sentence[-200:]
            citations.append((sentence, int(citation_num)))
        if not citations:
            pattern_old = r'"([^"+]+)"\s*\((?:[Pp]age|[Pp]\.? )\s*(\d+)\)'
            matches_old = re.findall(pattern_old, response_text)
            for idx, (quote, _page) in enumerate(matches_old, 1):
                citations.append((quote, idx))
        return citations

    # ------------------------------------------------------------------
    # Document source management
    # ------------------------------------------------------------------
    def get_document_source(self, conversation_id: str) -> dict | None:
        """Get document source info for a conversation."""
        return self._document_sources.get(conversation_id)

    def set_document_source(self, conversation_id: str, source_info: dict) -> None:
        """Set document source info for a conversation."""
        self._document_sources[conversation_id] = source_info

    def update_document_source(self, conversation_id: str, updates: dict) -> None:
        """Update document source info for a conversation.

        Parameters
        ----------
        conversation_id : str
            The conversation identifier
        updates : dict
            Dictionary of fields to update
        """
        if conversation_id in self._document_sources:
            self._document_sources[conversation_id].update(updates)
        else:
            self._document_sources[conversation_id] = updates

    @property
    def document_source_store(self) -> dict[str, dict]:
        """Get the document sources dict (for backward compatibility)."""
        return self._document_sources

    def clear_conversation_cache(self, conversation_id: str) -> None:
        """Clear all cached data for a conversation."""
        self._pdf_text_blocks.pop(conversation_id, None)
        self._snowflake_document_pages.pop(conversation_id, None)
        self._llm_referenced_quotes.pop(conversation_id, None)
        self._document_sources.pop(conversation_id, None)

    def clear_all_caches(self) -> None:
        """Clear all conversation caches."""
        self._pdf_text_blocks.clear()
        self._snowflake_document_pages.clear()
        self._llm_referenced_quotes.clear()
        self._document_sources.clear()

    # ------------------------------------------------------------------
    # Snowflake upload and parsing
    # ------------------------------------------------------------------
    async def upload_pdf_bytes_to_snowflake(
        self,
        client,
        pdf_bytes: bytes,
        filename: str,
        conversation_id: str,
        embed_images: bool = False,
    ) -> tuple[bool, str, str | None]:
        """Upload PDF bytes to Snowflake stage and trigger parsing.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        pdf_bytes : bytes
            Raw PDF file bytes
        filename : str
            Desired filename in stage
        conversation_id : str
            The conversation identifier
        embed_images : bool, optional
            Whether to extract and embed images, by default False

        Returns
        -------
        tuple[bool, str, str | None]
            Tuple of (success, message, stage_path)
        """
        import shutil
        import tempfile
        from pathlib import Path

        try:
            db_name = "OPENBB_AGENTS"
            user_schema = await self._get_user_schema(client)
            stage_name = "CORTEX_UPLOADS"

            try:
                await self._run_in_thread(
                    client.execute_statement, f"CREATE DATABASE IF NOT EXISTS {db_name}"
                )
            except Exception as e:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    logger.debug("Could not create database: %s", e)

            try:
                await self._run_in_thread(
                    client.execute_statement,
                    f"CREATE SCHEMA IF NOT EXISTS {db_name}.{user_schema}",
                )
            except Exception as e:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    logger.debug("Could not create schema: %s", e)

            qualified_stage_name = f'"{db_name}"."{user_schema}"."{stage_name}"'

            try:
                create_stage_query = f"""
                CREATE STAGE IF NOT EXISTS {qualified_stage_name}
                DIRECTORY = (ENABLE = TRUE)
                ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')
                """
                await self._run_in_thread(client.execute_statement, create_stage_query)
                await self._run_in_thread(
                    client.execute_statement,
                    f"ALTER STAGE {qualified_stage_name} REFRESH",
                )
            except Exception as e:
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    logger.debug("Could not ensure stage: %s", e)

            temp_dir = None
            try:
                temp_dir = tempfile.mkdtemp(prefix="snowflake_pdf_upload_")
                temp_file_path = Path(temp_dir) / filename

                with open(temp_file_path, "wb") as f:
                    f.write(pdf_bytes)

                await self._run_in_thread(
                    client.upload_file_to_stage,
                    str(temp_file_path),
                    stage_name,
                )

                stage_path = f"@{db_name}.{user_schema}.{stage_name}/{filename}"

                self.set_document_source(
                    conversation_id,
                    {
                        "source_type": "snowflake",
                        "file_name": filename,
                        "stage_path": stage_path,
                        "database": db_name,
                        "schema": user_schema,
                    },
                )

                asyncio.create_task(
                    self._parse_document_and_store_pages(
                        client,
                        stage_path,
                        filename,
                        conversation_id,
                        db_name,
                        user_schema,
                        embed_images=embed_images,
                        pdf_bytes=pdf_bytes,
                    )
                )

                return True, f"PDF uploaded to {stage_path}", stage_path

            finally:
                if temp_dir:
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

        except Exception as e:
            error_msg = f"Failed to upload PDF to Snowflake: {e}"
            logger.error("%s", error_msg, exc_info=True)
            return False, error_msg, None

    async def _parse_document_and_store_pages(
        self,
        client,
        stage_path: str,
        filename: str,
        conversation_id: str,
        target_database: str,
        target_schema: str,
        embed_images: bool = False,
        pdf_bytes: bytes | None = None,
    ) -> None:
        """Background task to parse document and store pages for citation matching.

        Uses Snowflake AI_PARSE_DOCUMENT to extract page content and optionally
        generates embeddings for semantic search.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        stage_path : str
            The stage path to the document
        filename : str
            The document filename
        conversation_id : str
            The conversation identifier
        target_database : str
            Database name for results table
        target_schema : str
            Schema name for results table
        embed_images : bool, optional
            Whether to extract and embed images, by default False
        pdf_bytes : bytes | None, optional
            Raw PDF bytes for image extraction, by default None
        """
        try:
            qualified_table = (
                f'"{target_database}"."{target_schema}"."DOCUMENT_PARSE_RESULTS"'
            )
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                FILE_NAME STRING,
                STAGE_PATH STRING,
                PAGE_NUMBER INTEGER,
                PAGE_CONTENT STRING,
                METADATA VARIANT,
                PARSED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            await self._run_in_thread(client.execute_statement, create_table_query)

            # Delete any existing records for this document to avoid duplicates
            escaped_filename = filename.replace("'", "''")
            escaped_stage_path = stage_path.replace("'", "''")
            delete_query = f"""
            DELETE FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}' OR STAGE_PATH = '{escaped_stage_path}'
            """
            await self._run_in_thread(client.execute_statement, delete_query)
            logger.debug("Deleted existing parse results for %s", filename)

            clean_path = stage_path.lstrip("@")
            if "/" in clean_path:
                parts = clean_path.split("/", 1)
                stage_name_extracted = f"@{parts[0]}"
                relative_file_path = parts[1]
            else:
                stage_name_extracted = f"@{clean_path}"
                relative_file_path = filename

            insert_query = f"""
            INSERT INTO {qualified_table} (FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, METADATA)
            WITH DOC AS (
              SELECT AI_PARSE_DOCUMENT(TO_FILE('{stage_name_extracted}', '{relative_file_path}'), {{'mode': 'LAYOUT', 'page_split': true}}) AS RAW
            )
            SELECT 
                '{filename}', 
                '{stage_path}', 
                INDEX + 1,
                GET(VALUE, 'content')::STRING,
                GET(DOC.RAW, 'metadata')
            FROM DOC, TABLE(FLATTEN(input => COALESCE(GET(DOC.RAW, 'pages'), DOC.RAW)))
            """
            await self._run_in_thread(client.execute_statement, insert_query)

            select_query = f"""
            SELECT PAGE_NUMBER, PAGE_CONTENT 
            FROM {qualified_table} 
            WHERE FILE_NAME = '{filename}' AND STAGE_PATH = '{stage_path}'
            ORDER BY PAGE_NUMBER
            """
            result_json = await self._run_in_thread(
                client.execute_statement, select_query
            )
            rows = json.loads(result_json) if result_json else []

            pages = []
            for row in rows:
                page_num = self._get_row_value(row, "PAGE_NUMBER", "page_number")
                content = self._get_row_value(row, "PAGE_CONTENT", "page_content")
                if page_num and content:
                    pages.append(
                        {
                            "page": int(page_num),
                            "content": content,
                            "file_name": filename,
                        }
                    )

            # Extract and upload images, inject references into page content
            if embed_images and pdf_bytes:
                user_schema = await self._get_user_schema(client)
                stage_name = "DOCUMENT_IMAGES"
                qualified_stage = f"OPENBB_AGENTS.{user_schema}.{stage_name}"

                # Ensure document images stage exists
                await self._create_document_images_stage(client)

                # Clean up old images for this document
                try:
                    filename_escaped = filename.replace("'", "''")
                    remove_sql = f"REMOVE @{qualified_stage}/{filename_escaped}/"
                    await self._run_in_thread(client.execute_statement, remove_sql)
                except Exception:
                    pass  # May be empty

                # Extract images from PDF
                images = self._extract_images_from_pdf(pdf_bytes)
                logger.info(
                    "Extracted %d images from %s for embedding",
                    len(images),
                    filename,
                )

                # Build page_number -> list of uploaded image paths
                page_images: dict[int, list[str]] = {}

                for img in images:
                    try:
                        # Simple path: filename/page_X_image_Y.format
                        img_filename = f"{filename}/page_{img['page_number']}_image_{img['image_index']}.{img['format']}"

                        await self._run_in_thread(
                            client.upload_bytes_to_stage,
                            img["image_bytes"],
                            img_filename,
                            stage_name,
                        )

                        # Store the stage path for this page
                        page_num = img["page_number"]
                        if page_num not in page_images:
                            page_images[page_num] = []
                        page_images[page_num].append(
                            f"@{qualified_stage}/{img_filename}"
                        )

                        logger.debug(
                            "Uploaded image %s for page %d", img_filename, page_num
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to upload image for page %d: %s",
                            img["page_number"],
                            e,
                        )
                        continue

                # Inject image references into page content
                for page_data in pages:
                    page_num = page_data["page"]
                    if page_num in page_images:
                        # Build markdown image references
                        img_refs = []
                        for idx, img_path in enumerate(page_images[page_num]):
                            img_refs.append(
                                f"\n\n![Figure {idx + 1} on page {page_num}]({img_path})"
                            )

                        # Append image references to end of page content
                        if img_refs:
                            page_data[
                                "content"
                            ] += "\n\n---\n**Embedded Images:**" + "".join(img_refs)

                            # Update the stored content in Snowflake
                            content_escaped = page_data["content"].replace("'", "''")
                            update_sql = f"""
                            UPDATE {qualified_table}
                            SET PAGE_CONTENT = '{content_escaped}'
                            WHERE FILE_NAME = '{filename}' 
                              AND STAGE_PATH = '{stage_path}'
                              AND PAGE_NUMBER = {page_num}
                            """
                            await self._run_in_thread(
                                client.execute_statement, update_sql
                            )
                            logger.debug(
                                "Updated page %d content with %d image references",
                                page_num,
                                len(img_refs),
                            )

            self.set_document_pages(conversation_id, pages)

            # Generate embeddings for semantic search
            if pages:
                logger.info(
                    "Generating embeddings for document %s (embed_images=%s)",
                    filename,
                    embed_images,
                )
                embedding_success = await self._generate_embeddings_for_document(
                    client=client,
                    file_name=filename,
                    stage_path=stage_path,
                    pages=pages,
                    pdf_bytes=pdf_bytes,
                    embed_images=embed_images,
                )
                if embedding_success:
                    logger.info("Embeddings generated successfully for %s", filename)
                else:
                    logger.warning("Failed to generate embeddings for %s", filename)

        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("Background parsing failed for %s: %s", filename, e)
                traceback.print_exc()

    async def check_existing_snowflake_document(
        self,
        client,
        pdf_bytes: bytes,
        known_filename: str | None = None,
    ) -> dict | None:
        """Check if PDF already exists in Snowflake DOCUMENT_PARSE_RESULTS.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        pdf_bytes : bytes
            Raw PDF bytes for hash generation
        known_filename : str | None, optional
            Optional known filename to search for

        Returns
        -------
        dict | None
            Dictionary with exists, file_name, pages, etc. or None if not found
        """
        try:
            if known_filename:
                file_name = known_filename

            else:
                pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
                file_name = f"widget_pdf_{pdf_hash}.pdf"

            user_schema = await self._get_user_schema(client)
            qualified_table = (
                f'"OPENBB_AGENTS"."{user_schema}"."DOCUMENT_PARSE_RESULTS"'
            )

            escaped_filename = file_name.replace("'", "''")
            check_query = f"""
            SELECT FILE_NAME, STAGE_PATH, PAGE_NUMBER, PAGE_CONTENT, PARSED_AT
            FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}'
            ORDER BY PAGE_NUMBER
            """

            result_json = await self._run_in_thread(
                client.execute_statement, check_query
            )
            rows = json.loads(result_json) if result_json else []

            if not rows:
                return None

            pages = []
            stage_path = None
            for row in rows:
                page_num = self._get_row_value(row, "PAGE_NUMBER", "page_number")
                content = self._get_row_value(row, "PAGE_CONTENT", "page_content")
                if not stage_path:
                    stage_path = self._get_row_value(row, "STAGE_PATH", "stage_path")

                if page_num and content:
                    pages.append({"page": int(page_num), "content": content})

            if pages:
                pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
                return {
                    "exists": True,
                    "file_name": file_name,
                    "page_count": len(pages),
                    "pages": pages,
                    "stage_path": stage_path,
                    "pdf_hash": pdf_hash,
                }

            return None

        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.debug("Error checking existing Snowflake document: %s", e)
            return None

    def trigger_snowflake_upload_for_widget_pdf(
        self, client, conversation_id: str
    ) -> None:
        """Trigger background upload of widget PDF to Snowflake if available.

        Only uploads if document source is widget type and not already uploaded.
        Resets upload flag on failure for retry capability.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        conversation_id : str
            The conversation identifier
        """
        try:
            source_info = self.get_document_source(conversation_id)
            if not source_info:
                logger.debug(
                    "No document source found for conversation %s", conversation_id
                )
                return

            if source_info.get("source_type") != "widget":
                logger.debug(
                    "Document source type is %s, not 'widget' - skipping upload",
                    source_info.get("source_type"),
                )
                return

            pdf_bytes = source_info.get("pdf_bytes")
            if not pdf_bytes:
                logger.debug(
                    "No PDF bytes found in document source for conversation %s",
                    conversation_id,
                )
                return

            if source_info.get("snowflake_uploaded"):
                logger.debug(
                    "PDF already uploaded to Snowflake for conversation %s",
                    conversation_id,
                )
                return

            filename = source_info.get(
                "file_name", f"widget_pdf_{conversation_id[:8]}.pdf"
            )
            source_info["snowflake_uploaded"] = True
            logger.debug(
                "Triggering background upload of %s for conversation %s",
                filename,
                conversation_id,
            )

            async def do_upload():
                try:
                    success, message, stage_path = (
                        await self.upload_pdf_bytes_to_snowflake(
                            client, pdf_bytes, filename, conversation_id
                        )
                    )

                    if success and stage_path:
                        source_info["stage_path"] = stage_path
                        source_info["snowflake_source_type"] = "snowflake"
                        logger.debug(
                            "Successfully uploaded %s to %s", filename, stage_path
                        )
                    else:
                        logger.error("Failed to upload %s: %s", filename, message)
                        # Reset flag on failure so it can be retried
                        source_info["snowflake_uploaded"] = False
                except Exception as upload_error:
                    logger.error(
                        "Exception during background PDF upload for %s: %s",
                        filename,
                        upload_error,
                        exc_info=True,
                    )
                    # Reset flag on exception so it can be retried
                    source_info["snowflake_uploaded"] = False

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(do_upload())
                logger.debug("Created async task for PDF upload")
            except RuntimeError:
                # No event loop running, use thread instead
                logger.debug("No event loop found, using thread for PDF upload")

                def run_async():
                    try:
                        asyncio.run(do_upload())
                    except Exception as thread_error:
                        logger.error(
                            "Exception in upload thread for %s: %s",
                            filename,
                            thread_error,
                            exc_info=True,
                        )

                thread = threading.Thread(target=run_async, daemon=True)
                thread.start()

        except Exception as e:
            logger.error(
                "Unexpected error in trigger_snowflake_upload_for_widget_pdf for conversation %s: %s",
                conversation_id,
                e,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Widget data handling
    # ------------------------------------------------------------------
    async def handle_get_widget_data_tool_call(
        self,
        tool_args: dict,
        widgets_primary: list | None,
        widgets_secondary: list | None,
        client,
        conversation_id: str,
    ) -> tuple[str, dict]:
        """
        Handle get_widget_data tool calls from the LLM.

        This intercepts tool calls for get_widget_data to:
        1. Check if the document is already parsed in Snowflake (by filename)
        2. If parsed, return the content directly without fetching from widget
        3. Otherwise, look up widget by UUID and fetch data

        Args:
            tool_args: Arguments from the tool call (should contain widget_uuids)
            widgets_primary: List of primary widgets from request.widgets
            widgets_secondary: List of secondary widgets from request.widgets
            client: The Snowflake client
            conversation_id: The conversation ID

        Returns:
            Tuple of (llm_output, raw_data) like other tool executors
        """
        from openbb_ai import get_widget_data
        from openbb_ai.models import WidgetRequest

        # Extract widget UUIDs from tool args
        widget_uuids = tool_args.get("widget_uuids", [])
        if isinstance(widget_uuids, str):
            widget_uuids = [widget_uuids]

        if not widget_uuids:
            # If no specific UUIDs, use all primary widgets
            widget_uuids = []
            if widgets_primary:
                widget_uuids.extend([str(w.uuid) for w in widgets_primary])

        # Build a lookup of available widgets
        all_widgets = []
        if widgets_primary:
            all_widgets.extend(widgets_primary)
        if widgets_secondary:
            all_widgets.extend(widgets_secondary)

        widget_lookup = {str(w.uuid): w for w in all_widgets}

        # Find requested widgets
        matched_widgets = []
        for uuid_str in widget_uuids:
            if uuid_str in widget_lookup:
                matched_widgets.append(widget_lookup[uuid_str])

        if not matched_widgets:
            return "No matching widgets found for the requested UUIDs.", {
                "error": "No widgets found"
            }

        # Check if any of the matched widgets reference documents already parsed in Snowflake
        # If so, we can return that content directly without fetching from widget
        if client:
            for widget in matched_widgets:
                known_filename = self.extract_filename_from_widget(widget)
                if known_filename:
                    try:
                        # Query DOCUMENT_PARSE_RESULTS to see if file exists
                        db_name = "OPENBB_AGENTS"
                        snowflake_user = await asyncio.to_thread(
                            client.get_current_user
                        )
                        sanitized_user = "".join(
                            c if c.isalnum() else "_" for c in snowflake_user
                        )
                        schema_name = f"USER_{sanitized_user}".upper()
                        qualified_table = (
                            f'"{db_name}"."{schema_name}"."DOCUMENT_PARSE_RESULTS"'
                        )

                        escaped_filename = known_filename.replace("'", "''")
                        query = f"""
                        SELECT FILE_NAME, PAGE_NUMBER, PAGE_CONTENT
                        FROM {qualified_table}
                        WHERE FILE_NAME = '{escaped_filename}'
                        ORDER BY PAGE_NUMBER
                        """

                        result_json = await asyncio.to_thread(
                            client.execute_statement, query
                        )
                        rows = json.loads(result_json) if result_json else []

                        if rows:
                            # Document is already parsed - return content directly

                            # Build content from parsed pages
                            content_parts = []
                            for row in rows:
                                page_num = self._get_row_value(
                                    row, "PAGE_NUMBER", "page_number"
                                )
                                content = self._get_row_value(
                                    row, "PAGE_CONTENT", "page_content"
                                )
                                if content:
                                    content_parts.append(
                                        f"[Page {page_num}]\n{content}"
                                    )

                            parsed_content = "\n\n".join(content_parts)

                            # Store in snowflake_document_pages for citation matching
                            self._snowflake_document_pages[conversation_id] = [
                                {
                                    "page": int(
                                        self._get_row_value(
                                            row, "PAGE_NUMBER", "page_number"
                                        )
                                        or 0
                                    ),
                                    "content": self._get_row_value(
                                        row, "PAGE_CONTENT", "page_content"
                                    ),
                                    "file_name": known_filename,
                                }
                                for row in rows
                            ]

                            # Store document source info
                            self._document_sources[conversation_id] = {
                                "source_type": "snowflake",
                                "file_name": known_filename,
                                "has_snowflake_content": True,
                                "snowflake_page_count": len(rows),
                            }

                            return (
                                f"Document '{known_filename}' content ({len(rows)} pages):\n\n{parsed_content}",
                                {
                                    "source": "snowflake_cache",
                                    "file_name": known_filename,
                                    "page_count": len(rows),
                                },
                            )
                    except Exception as e:
                        logger.debug(
                            "Error checking Snowflake for %s: %s", known_filename, e
                        )
                        # Continue to fetch from widget

        # Build widget requests
        widget_requests = []
        for widget in matched_widgets:
            input_args = {}
            if hasattr(widget, "params") and widget.params:
                input_args = {p.name: p.current_value for p in widget.params}

            widget_req = WidgetRequest(
                widget=widget,
                input_arguments=input_args,
            )
            widget_requests.append(widget_req)

        # Get widget data request object (this is what tells the UI to fetch data)
        widget_data_request = get_widget_data(widget_requests)

        # Store widget info for later processing when data comes back
        # The actual data will come back in a subsequent request as a tool message
        widget_info = {
            "widget_uuids": widget_uuids,
            "widget_names": [getattr(w, "name", str(w.uuid)) for w in matched_widgets],
            "request_sent": True,
        }

        # Return the widget data request for the UI to execute
        # This will trigger the UI to fetch the actual data
        output = f"Requesting data for {len(matched_widgets)} widget(s): {', '.join(widget_info['widget_names'])}"

        return output, {
            "widget_data_request": widget_data_request.model_dump(),
            "widget_info": widget_info,
        }

    async def store_widget_data_in_snowflake(
        self,
        client,
        widget_uuid: str,
        widget_name: str,
        data_content: str | dict | list,
        conversation_id: str,
        data_type: str = "json",
    ) -> tuple[bool, str]:
        """Store widget data in Snowflake for reference and future queries.

        Creates WIDGET_DATA_CACHE table if needed and upserts the data.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        widget_uuid : str
            The widget's UUID
        widget_name : str
            The widget's name
        data_content : str | dict | list
            The data content (JSON, text, etc.)
        conversation_id : str
            The conversation ID
        data_type : str, optional
            Type of data ("json", "pdf", "text"), by default "json"

        Returns
        -------
        tuple[bool, str]
            Tuple of (success, message)
        """
        try:
            # Get user-specific schema
            db_name = "OPENBB_AGENTS"
            snowflake_user = await asyncio.to_thread(client.get_current_user)
            sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
            schema_name = f"USER_{sanitized_user}".upper()

            qualified_table = f"{db_name}.{schema_name}.WIDGET_DATA_CACHE"

            # Create table if not exists
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {qualified_table} (
                WIDGET_UUID STRING,
                WIDGET_NAME STRING,
                CONVERSATION_ID STRING,
                DATA_TYPE STRING,
                DATA_CONTENT VARIANT,
                CREATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
                UPDATED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """
            await asyncio.to_thread(client.execute_statement, create_table_query)

            # Serialize data content
            if isinstance(data_content, (dict, list)):
                data_json = json.dumps(data_content)
            else:
                data_json = json.dumps({"content": str(data_content)})

            # Escape for SQL
            escaped_uuid = widget_uuid.replace("'", "''")
            escaped_name = widget_name.replace("'", "''")
            escaped_conv_id = conversation_id.replace("'", "''")
            escaped_data = data_json.replace("'", "''")

            # Upsert (merge) the data
            merge_query = f"""
            MERGE INTO {qualified_table} AS target
            USING (SELECT '{escaped_uuid}' AS uuid, '{escaped_conv_id}' AS conv_id) AS source
            ON target.WIDGET_UUID = source.uuid AND target.CONVERSATION_ID = source.conv_id
            WHEN MATCHED THEN
                UPDATE SET DATA_CONTENT = PARSE_JSON('{escaped_data}'), 
                           DATA_TYPE = '{data_type}',
                           UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (WIDGET_UUID, WIDGET_NAME, CONVERSATION_ID, DATA_TYPE, DATA_CONTENT)
                VALUES ('{escaped_uuid}', '{escaped_name}', '{escaped_conv_id}', '{data_type}', PARSE_JSON('{escaped_data}'))
            """
            await asyncio.to_thread(client.execute_statement, merge_query)

            return True, "Widget data stored in Snowflake"

        except Exception as e:
            error_msg = f"Failed to store widget data: {e}"
            logger.error("%s", error_msg)
            return False, error_msg

    def _check_existing_snowflake_document_sync(
        self, client, pdf_bytes: bytes, known_filename: str | None = None
    ) -> dict | None:
        """Synchronous wrapper for check_existing_snowflake_document.

        Handles event loop detection and runs async method appropriately.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        pdf_bytes : bytes
            Raw PDF bytes
        known_filename : str | None, optional
            Optional known filename

        Returns
        -------
        dict | None
            Same as check_existing_snowflake_document
        """
        # Try to run in existing loop or create new one
        try:
            loop = asyncio.get_running_loop()
            # We're in async context, create task
            return asyncio.run_coroutine_threadsafe(
                self.check_existing_snowflake_document(
                    client, pdf_bytes, known_filename
                ),
                loop,
            ).result()
        except RuntimeError:
            # No running loop, run directly
            return asyncio.run(
                self.check_existing_snowflake_document(
                    client, pdf_bytes, known_filename
                )
            )

    # ------------------------------------------------------------------
    # Document orchestration methods
    # ------------------------------------------------------------------
    async def prepare_document_widgets(
        self,
        widgets_primary: list | None,
        widgets_secondary: list | None,
        conversation_id: str,
        client,
    ) -> dict[str, Any]:
        """Prepare document widgets by checking if documents are ready and extracting PDF positions.

        This handles all document-specific logic for widget preparation including:
        - Identifying document widgets from primary/secondary widget lists
        - Checking if documents are already parsed in Snowflake
        - Fetching PDF bytes for citation bounding boxes
        - Loading PDF positions and document pages for citation matching
        - Building widget context metadata

        Parameters
        ----------
        widgets_primary : list | None
            List of primary widgets
        widgets_secondary : list | None
            List of secondary widgets
        conversation_id : str
            The conversation identifier
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        dict[str, Any]
            Dict with keys:
            - 'is_document': bool - whether any document widgets were found
            - 'widget_context_str': str - formatted context string for LLM
            - 'widget_for_citations': widget object or None
            - 'widget_input_args_for_citations': dict or None
            - 'widget_context_metadata': dict or None
            - 'stage_path': str or None
        """
        from .widget_handler import WidgetHandler

        result = {
            "is_document": False,
            "widget_context_str": "",
            "widget_for_citations": None,
            "widget_input_args_for_citations": None,
            "widget_context_metadata": None,
            "stage_path": None,
        }

        # Check all widgets (primary and secondary)
        all_widgets = list(widgets_primary or []) + list(widgets_secondary or [])
        if not all_widgets:
            return result

        widget_handler = await WidgetHandler.instance()

        # Find document widgets
        document_widgets = [
            w for w in all_widgets if widget_handler.is_document_widget(w)
        ]

        if not document_widgets:
            return result

        # For now, handle the first document widget (can be extended for multiple)
        widget = document_widgets[0]
        result["is_document"] = True

        # Extract document paths
        document_paths = widget_handler.extract_document_paths(widget)
        if not document_paths:
            return result

        file_path_in_widget = document_paths[0]
        if not file_path_in_widget or not file_path_in_widget.startswith("@"):
            return result

        result["stage_path"] = file_path_in_widget

        # Extract filename and check if document is ready
        filename_in_widget = self.extract_filename_from_stage_path(file_path_in_widget)
        if not filename_in_widget:
            return result

        available_docs = await self._run_in_thread(client.list_cortex_documents)
        doc_is_ready = any(
            doc[0] == filename_in_widget and doc[2] for doc in available_docs
        )

        if not doc_is_ready:
            return result

        # Note: PDF positions will be loaded from widget data when it comes back as a tool message
        # We don't call get_widget_data here because it returns a FunctionCallSSE that must be
        # yielded to the client, not used directly.

        # Build widget context metadata
        widget_label = getattr(widget, "name", None) or "Cortex Documents"
        document_label = filename_in_widget
        stage_path = file_path_in_widget

        result["widget_for_citations"] = widget
        result["widget_context_metadata"] = {
            "widget_label": widget_label,
            "widget_uuid": str(widget.uuid),
            "document_label": document_label,
            "stage_path": stage_path,
        }

        result["widget_input_args_for_citations"] = {
            "conversation_id": conversation_id,
            "stage_path": stage_path,
            "widget_label": widget_label,
            "file_name": filename_in_widget,
            "widget_uuid": str(widget.uuid),
            "widget_title": getattr(widget, "title", None)
            or getattr(widget, "name", None),
        }

        # Build widget context string
        metadata_lines = [
            "The user explicitly selected this document from the widget. Do NOT ask which document; cite this source directly.",
            f"Widget Name: {widget_label}",
            f"Widget UUID: {widget.uuid}",
            f"Document/File: {document_label}",
            f"Stage Path: {stage_path}",
        ]

        result["widget_context_str"] = (
            "\n".join(metadata_lines)
            + f"\n\n--- Widget Document: {document_label} ---\n"
            + "Document is already parsed in Snowflake and ready for queries.\n"
            + f"Location: OPENBB_AGENTS.USER_{{username}}.DOCUMENT_PARSE_RESULTS WHERE FILE_NAME = '{filename_in_widget}'\n"
            + "------\n"
        )

        # ALWAYS load PDF metadata first - this gives the LLM document structure
        # The metadata (outline, TOC, page summaries) helps the model navigate
        pdf_bytes = None
        if filename_in_widget.lower().endswith(".pdf"):
            doc_source = self.get_document_source(conversation_id) or {}
            cached_pdf_bytes = doc_source.get("widget_pdf_bytes")
            if cached_pdf_bytes:
                pdf_bytes = cached_pdf_bytes

            # Check what we already have cached
            has_metadata = self.get_pdf_metadata(conversation_id) is not None
            has_positions = self.get_pdf_blocks(conversation_id) is not None

            # Try loading from Snowflake first (fast, no PDF download)
            if not has_metadata:
                if await self.load_document_metadata_from_snowflake(
                    client, conversation_id, filename_in_widget
                ):
                    has_metadata = True
                    logger.debug(
                        "Loaded metadata from Snowflake for %s", filename_in_widget
                    )

            if not has_positions:
                if await self.load_pdf_positions_from_snowflake(
                    client, conversation_id, filename_in_widget, stage_path
                ):
                    has_positions = True
                    logger.debug(
                        "Loaded positions from Snowflake for %s", filename_in_widget
                    )

            # If we're missing either, download PDF and extract both in one pass
            if not has_metadata or not has_positions:
                try:
                    if pdf_bytes is None:
                        pdf_base64 = await self._run_in_thread(
                            client.download_cortex_document,
                            filename_in_widget,
                        )
                        pdf_bytes = base64.b64decode(pdf_base64)
                        doc_source["widget_pdf_bytes"] = pdf_bytes
                        self.set_document_source(conversation_id, doc_source)

                    # Single extraction pass for positions + metadata
                    need_metadata = not has_metadata
                    _, text_positions, pdf_metadata = self.extract_pdf_with_positions(
                        pdf_bytes, extract_metadata=need_metadata
                    )

                    # Store positions if we needed them
                    if not has_positions and text_positions:
                        self.set_pdf_blocks(conversation_id, text_positions)
                        await self.store_pdf_positions_in_snowflake(
                            client, filename_in_widget, stage_path, text_positions
                        )
                        logger.info(
                            "Extracted %d text positions for %s",
                            len(text_positions),
                            filename_in_widget,
                        )

                    # Store metadata if we needed it
                    if need_metadata and pdf_metadata:
                        self.set_pdf_metadata(conversation_id, pdf_metadata)
                        await self.store_document_metadata(
                            client, filename_in_widget, stage_path, pdf_metadata
                        )
                        logger.info(
                            "Extracted metadata for %s: %d pages, %d outline entries",
                            filename_in_widget,
                            pdf_metadata.get("page_count", 0),
                            len(pdf_metadata.get("outline", [])),
                        )
                except Exception as e:
                    logger.warning("Failed to extract PDF data: %s", e)

        # Load document pages for citation matching
        try:
            await self.load_snowflake_document_pages(
                client, conversation_id, filename_in_widget
            )
        except Exception as exc:
            logger.debug("Unexpected error loading Snowflake document pages: %s", exc)

        return result

    async def ensure_document_ready(
        self,
        widget,
        conversation_id: str,
        client,
    ) -> dict[str, Any]:
        """Ensure a document widget's document is ready for querying.

        Loads PDF positions and document pages if available.

        Parameters
        ----------
        widget : Widget
            The document widget
        conversation_id : str
            The conversation identifier
        client : SnowflakeClient
            The Snowflake client

        Returns
        -------
        dict[str, Any]
            Dictionary with widget_for_citations, citations_required, context strings, etc.
        """
        stage_path = self.extract_document_stage_path(widget)
        if not stage_path:
            return {}

        filename = self.extract_filename_from_stage_path(stage_path)
        if not filename:
            return {}

        available_docs = await self._run_in_thread(client.list_cortex_documents)
        doc_is_ready = any(doc[0] == filename and doc[2] for doc in available_docs)

        if not doc_is_ready:
            return {}

        # Note: get_widget_data returns a FunctionCallSSE that must be yielded to client.
        # The actual widget data will be processed when it comes back as a tool message.
        # This method only ensures the document is ready in Snowflake.

        widget_label = getattr(widget, "name", None) or "Cortex Documents"
        widget_context_metadata = {
            "widget_label": widget_label,
            "widget_uuid": str(widget.uuid),
            "document_label": filename,
            "stage_path": stage_path,
        }

        widget_input_args_for_citations = {
            "conversation_id": conversation_id,
            "stage_path": stage_path,
            "widget_label": widget_label,
            "file_name": filename,
            "widget_uuid": str(widget.uuid),
            "widget_title": getattr(widget, "title", None)
            or getattr(widget, "name", None),
        }

        metadata_lines = [
            "The user explicitly selected this document from the widget. Do NOT ask which document; cite this source directly.",
            f"Widget Name: {widget_label}",
            f"Widget UUID: {widget.uuid}",
            f"Document/File: {filename}",
            f"Stage Path: {stage_path}",
        ]

        widget_context_str = (
            "\n".join(metadata_lines)
            + f"\n\n--- Widget Document: {filename} ---\n"
            + "Document is already parsed in Snowflake and ready for queries.\n"
            + f"Location: OPENBB_AGENTS.USER_{{username}}.DOCUMENT_PARSE_RESULTS WHERE FILE_NAME = '{filename}'\n"
            + "------\n"
        )

        try:
            positions_loaded = await self.load_pdf_positions_from_snowflake(
                client, conversation_id, filename, stage_path
            )
            if not positions_loaded and filename.lower().endswith(".pdf"):
                try:
                    # Check document source cache first
                    doc_source = self.get_document_source(conversation_id) or {}
                    pdf_bytes = doc_source.get("widget_pdf_bytes")
                    if pdf_bytes:
                        logger.debug(
                            "Using cached PDF bytes in ensure_document_ready (%d bytes)",
                            len(pdf_bytes),
                        )
                    else:
                        logger.debug(
                            "Downloading PDF %s in ensure_document_ready (no cache)",
                            filename,
                        )
                        pdf_base64 = await self._run_in_thread(
                            client.download_cortex_document,
                            filename,
                        )
                        pdf_bytes = base64.b64decode(pdf_base64)
                        # Cache for future use
                        doc_source["widget_pdf_bytes"] = pdf_bytes
                        self.set_document_source(conversation_id, doc_source)

                    _, text_positions, _ = self.extract_pdf_with_positions(pdf_bytes)
                    await self.store_pdf_positions_in_snowflake(
                        client,
                        filename,
                        stage_path,
                        text_positions,
                    )
                    self.set_pdf_blocks(conversation_id, text_positions)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            await self.load_snowflake_document_pages(client, conversation_id, filename)
        except Exception:
            pass

        return {
            "widget_for_citations": widget,
            "citations_required": True,
            "widget_context_str": widget_context_str,
            "widget_context_metadata": widget_context_metadata,
            "widget_input_args_for_citations": widget_input_args_for_citations,
        }

    # ------------------------------------------------------------------
    # Stage file management
    # ------------------------------------------------------------------
    async def remove_file_from_stage(
        self,
        client,
        file_path: str,
    ) -> tuple[bool, str]:
        """Remove a file from Snowflake stage.

        Also removes corresponding entries from DOCUMENT_PARSE_RESULTS.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_path : str
            The file path to remove. Can be:
            - Full stage path: "@OPENBB_AGENTS.USER_DLEE.CORTEX_UPLOADS/filename.pdf.gz"
            - Short form: "filename.pdf.gz" (will use user's default stage)
            - Relative path: "CORTEX_UPLOADS/filename.pdf.gz"

        Returns
        -------
        tuple[bool, str]
            Tuple of (success, message)
        """
        try:
            # Get user-specific schema
            db_name = "OPENBB_AGENTS"
            snowflake_user = await self._run_in_thread(client.get_current_user)
            sanitized_user = "".join(c if c.isalnum() else "_" for c in snowflake_user)
            schema_name = f"USER_{sanitized_user}".upper()
            stage_name = "CORTEX_UPLOADS"

            # Normalize the file path
            if file_path.startswith("@"):
                # Full stage path provided
                stage_path = file_path
            elif "/" in file_path:
                # Relative path with stage name
                stage_path = f"@{db_name}.{schema_name}.{file_path}"
            else:
                # Just filename
                stage_path = f"@{db_name}.{schema_name}.{stage_name}/{file_path}"

            # Execute REMOVE command - quote the path to handle filenames with spaces
            remove_query = f"REMOVE '{stage_path}'"

            try:
                _ = await self._run_in_thread(client.execute_statement, remove_query)
            except Exception as e:
                error_msg = f"Failed to remove file from stage: {e}"
                if os.environ.get("SNOWFLAKE_DEBUG"):
                    logger.error("%s", error_msg)
                    traceback.print_exc()
                return False, error_msg

            # Also delete from DOCUMENT_PARSE_RESULTS if exists
            # Extract filename from the path
            filename = stage_path.split("/")[-1]
            # Remove .gz extension if present for matching
            base_filename = filename.rstrip(".gz")

            qualified_table = f'"{db_name}"."{schema_name}"."DOCUMENT_PARSE_RESULTS"'
            escaped_filename = base_filename.replace("'", "''")
            escaped_filename_gz = filename.replace("'", "''")

            delete_query = f"""
            DELETE FROM {qualified_table}
            WHERE FILE_NAME = '{escaped_filename}' OR FILE_NAME = '{escaped_filename_gz}'
            """

            try:
                await self._run_in_thread(client.execute_statement, delete_query)

            except Exception:
                pass

            return True, f"File removed: {stage_path}"

        except Exception as e:
            error_msg = f"Failed to remove file from stage: {e}"
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.error("%s", error_msg)
                traceback.print_exc()

            return False, error_msg

    def remove_file_from_stage_sync(
        self,
        client,
        file_path: str,
    ) -> tuple[bool, str]:
        """Synchronous wrapper for remove_file_from_stage.

        Handles event loop detection and runs async method appropriately.

        Parameters
        ----------
        client : SnowflakeClient
            The Snowflake client
        file_path : str
            The file path to remove

        Returns
        -------
        tuple[bool, str]
            Same as remove_file_from_stage
        """
        try:
            loop = asyncio.get_running_loop()
            future = asyncio.run_coroutine_threadsafe(
                self.remove_file_from_stage(client, file_path),
                loop,
            )
            return future.result(timeout=30)
        except RuntimeError:
            return asyncio.run(self.remove_file_from_stage(client, file_path))
        except Exception as e:
            if os.environ.get("SNOWFLAKE_DEBUG"):
                logger.debug("Error in sync wrapper: %s", e)
            return False, str(e)
