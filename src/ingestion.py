"""
Document Ingestion Pipeline

Responsibilities:
- Validate uploaded files
- Dispatch to the appropriate parser
- Clean extracted text
- Generate metadata
- Return a standardized document object

Dependencies:
- parsers.py
- cleaner.py
- metadata.py
- models.py
"""

from pathlib import Path
from typing import BinaryIO, Optional

from src.parsers import parse_document
from src.cleaner import clean_text
from src.metadata import generate_metadata
from src.models import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails."""


class DocumentIngestor:
    """Handles document ingestion."""

    def ingest(
        self,
        file: BinaryIO,
        filename: str,
    ) -> Document:
        """
        Main ingestion pipeline.

        Steps:
            1. Validate file
            2. Parse text
            3. Clean text
            4. Generate metadata
            5. Return Document object
        """

        extension = Path(filename).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise DocumentIngestionError(
                f"Unsupported file type: {extension}"
            )

        pages = parse_document(file=file, extension=extension)

        if not pages:
            raise DocumentIngestionError(
                "No readable text found in the document."
            )

        cleaned_pages = []

        for page in pages:
            cleaned_text = clean_text(page["text"])

            cleaned_pages.append(
                {
                    "page_number": page["page_number"],
                    "text": cleaned_text,
                }
            )

        # Check whether the entire document is empty
        combined_text = "".join(page["text"] for page in cleaned_pages).strip()

        if not combined_text:
            raise DocumentIngestionError(
                "The document contains no readable text."
            )

        metadata = generate_metadata(
            filename=filename,
            extension=extension,
            pages=cleaned_pages,
        )

        return Document(
            metadata=metadata,
            pages=cleaned_pages,
        )