"""
test_retrieval.py

Integration tests for the semantic retrieval pipeline.
"""

import os

import pytest
from dotenv import load_dotenv

from src.chunk_models import Chunk
from src.embeddings import EmbeddingService
from src.indexing import DocumentIndexer
from src.retrieval import Retriever
from src.vectordb import VectorDatabase

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="GEMINI_API_KEY not configured."
)


def create_chunks():

    return [
        Chunk(
            chunk_id="chunk_001",
            document_id="doc001",
            filename="ai_guide.pdf",
            page_number=1,
            chunk_index=0,
            text="Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
            start_char=0,
            end_char=98,
        ),
        Chunk(
            chunk_id="chunk_002",
            document_id="doc001",
            filename="ai_guide.pdf",
            page_number=2,
            chunk_index=1,
            text="Machine Learning is a subset of Artificial Intelligence that learns from data.",
            start_char=99,
            end_char=182,
        ),
        Chunk(
            chunk_id="chunk_003",
            document_id="doc002",
            filename="rag.pdf",
            page_number=1,
            chunk_index=0,
            text="Retrieval-Augmented Generation combines vector search with a large language model.",
            start_char=0,
            end_char=88,
        ),
    ]


@pytest.fixture
def retriever():

    collection_name = "test_retrieval"

    vectordb = VectorDatabase(
        collection_name=collection_name
    )

    vectordb.reset()

    indexer = DocumentIndexer(
        vector_database=vectordb
    )

    indexer.index_chunks(
        create_chunks()
    )

    yield Retriever(
        vector_database=vectordb
    )

    vectordb.reset()


def test_retrieve_returns_results(retriever):

    results = retriever.retrieve(
        query="What is Artificial Intelligence?",
        top_k=3,
    )

    assert len(results) > 0


def test_retrieve_respects_top_k(retriever):

    results = retriever.retrieve(
        query="Artificial Intelligence",
        top_k=2,
    )

    assert len(results) <= 2


def test_retrieve_contains_metadata(retriever):

    results = retriever.retrieve(
        query="Machine Learning",
        top_k=1,
    )

    result = results[0]

    assert "filename" in result
    assert "page_number" in result
    assert "chunk_id" in result
    assert "document_id" in result
    assert "text" in result


def test_empty_query_returns_empty_list(retriever):

    results = retriever.retrieve(
        query="",
        top_k=5,
    )

    assert results == []


def test_context_generation(retriever):

    context = retriever.retrieve_context(
        query="Retrieval-Augmented Generation",
        top_k=2,
    )

    assert isinstance(context, str)
    assert len(context) > 0
    assert "Source:" in context


def test_similarity_search_returns_relevant_text(retriever):

    results = retriever.retrieve(
        query="Machine Learning",
        top_k=1,
    )

    assert (
        "Machine Learning" in results[0]["text"]
        or "Artificial Intelligence" in results[0]["text"]
    )


def test_no_results_database():

    vectordb = VectorDatabase(
        collection_name="empty_collection"
    )

    vectordb.reset()

    retriever = Retriever(
        vector_database=vectordb
    )

    results = retriever.retrieve(
        query="Quantum Computing",
        top_k=5,
    )

    assert isinstance(results, list)

    vectordb.reset()