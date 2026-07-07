"""
metadata.py

Generates metadata for ingested documents.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Dict, List


def generate_document_id(filename: str, pages: List[Dict]) -> str:
    """
    Generate a deterministic document ID based on filename and content.
    """

    hasher = hashlib.sha256()

    hasher.update(filename.encode("utf-8"))

    for page in pages:
        hasher.update(page["text"].encode("utf-8"))

    return hasher.hexdigest()


def detect_language() -> str:
    """
    Placeholder for language detection.

    This will be replaced in Phase 9 with langdetect.
    """

    return "unknown"


def generate_metadata(
    filename: str,
    extension: str,
    pages: List[Dict],
) -> Dict:
    """
    Generate metadata for an ingested document.
    """

    full_text = "\n".join(page["text"] for page in pages)

    metadata = {
        "document_id": generate_document_id(filename, pages),
        "filename": filename,
        "file_type": extension.replace(".", ""),
        "page_count": len(pages),
        "character_count": len(full_text),
        "word_count": len(full_text.split()),
        "language": detect_language(),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    return metadata