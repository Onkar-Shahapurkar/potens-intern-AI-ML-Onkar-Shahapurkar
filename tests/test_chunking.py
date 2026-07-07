"""
test_chunking.py

Tests for the document chunking pipeline.
"""

from src.chunking import DocumentChunker
from src.models import (
    Document,
    DocumentMetadata,
    Page,
)


def create_document(text: str) -> Document:
    """
    Helper function to create a test document.
    """

    metadata = DocumentMetadata(
        document_id="doc001",
        filename="sample.txt",
        file_type="txt",
        page_count=1,
        character_count=len(text),
        word_count=len(text.split()),
        language="en",
        uploaded_at="2026-01-01T00:00:00Z",
    )

    page = Page(
        page_number=1,
        text=text,
    )

    return Document(
        metadata=metadata,
        pages=[page],
    )


def test_small_document_single_chunk():
    """
    Small documents should produce one chunk.
    """

    document = create_document(
        "Artificial Intelligence is transforming industries."
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1
    assert chunks[0].page_number == 1
    assert chunks[0].document_id == "doc001"


def test_large_document_multiple_chunks():
    """
    Large documents should be split into multiple chunks.
    """

    text = "Machine Learning " * 500

    document = create_document(text)

    chunker = DocumentChunker(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1


def test_chunk_ids_are_unique():
    """
    Every generated chunk should have a unique ID.
    """

    text = "Deep Learning " * 500

    document = create_document(text)

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    ids = [chunk.chunk_id for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_chunk_metadata_preserved():
    """
    Chunk metadata should match the source document.
    """

    document = create_document(
        "Natural Language Processing " * 100
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    for chunk in chunks:

        assert chunk.document_id == "doc001"
        assert chunk.filename == "sample.txt"
        assert chunk.page_number == 1


def test_chunk_has_text():
    """
    Generated chunks should never be empty.
    """

    document = create_document(
        "Retrieval Augmented Generation " * 200
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    for chunk in chunks:

        assert chunk.text.strip() != ""


def test_chunk_character_offsets():
    """
    Character offsets should be valid.
    """

    document = create_document(
        "Artificial Intelligence " * 300
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    for chunk in chunks:

        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char


def test_chunk_length_within_limit():
    """
    Chunk length should not exceed configured size.
    """

    document = create_document(
        "AI " * 3000
    )

    chunker = DocumentChunker(
        chunk_size=600,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_document(document)

    for chunk in chunks:

        assert len(chunk.text) <= 600


def test_empty_document_returns_no_chunks():
    """
    Empty documents should produce no chunks.
    """

    document = create_document("")

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    assert chunks == []