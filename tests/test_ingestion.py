"""
test_ingestion.py

Basic tests for the document ingestion pipeline.
"""

from io import BytesIO

import pytest

from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)


def test_ingest_txt_document():
    """
    Should successfully ingest a TXT document.
    """

    ingestor = DocumentIngestor()

    content = b"Artificial Intelligence is transforming industries."

    document = ingestor.ingest(
        file=BytesIO(content),
        filename="sample.txt",
    )

    assert document.metadata.filename == "sample.txt"
    assert document.metadata.file_type == "txt"
    assert document.metadata.page_count == 1
    assert len(document.pages) == 1
    assert "Artificial Intelligence" in document.pages[0].text


def test_empty_txt_document():
    """
    Empty files should raise an exception.
    """

    ingestor = DocumentIngestor()

    with pytest.raises(DocumentIngestionError):

        ingestor.ingest(
            file=BytesIO(b""),
            filename="empty.txt",
        )


def test_unsupported_file_type():
    """
    Unsupported extensions should be rejected.
    """

    ingestor = DocumentIngestor()

    with pytest.raises(DocumentIngestionError):

        ingestor.ingest(
            file=BytesIO(b"hello"),
            filename="image.png",
        )


def test_document_metadata_exists():
    """
    Metadata should be generated correctly.
    """

    ingestor = DocumentIngestor()

    document = ingestor.ingest(
        file=BytesIO(
            b"Large Language Models are useful."
        ),
        filename="notes.txt",
    )

    metadata = document.metadata

    assert metadata.document_id
    assert metadata.word_count > 0
    assert metadata.character_count > 0
    assert metadata.language == "unknown"


def test_page_numbers_start_at_one():
    """
    First page number should always be 1.
    """

    ingestor = DocumentIngestor()

    document = ingestor.ingest(
        file=BytesIO(b"Hello World"),
        filename="page.txt",
    )

    assert document.pages[0].page_number == 1


def test_text_cleaning():
    """
    Text should be cleaned before returning.
    """

    ingestor = DocumentIngestor()

    raw = b"Hello     World\n\n\n\nThis    is    a    test."

    document = ingestor.ingest(
        file=BytesIO(raw),
        filename="clean.txt",
    )

    text = document.pages[0].text

    assert "     " not in text
    assert "\n\n\n" not in text