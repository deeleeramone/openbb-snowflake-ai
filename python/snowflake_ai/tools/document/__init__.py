"""Document tool handlers."""

from .read import handle_read_document
from .search import handle_search_document
from .images import handle_get_document_images
from .ocr import handle_ocr_image

__all__ = [
    "handle_read_document",
    "handle_search_document",
    "handle_get_document_images",
    "handle_ocr_image",
]
