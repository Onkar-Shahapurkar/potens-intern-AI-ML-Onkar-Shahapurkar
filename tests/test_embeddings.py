"""
test_embeddings.py

Integration tests for:

- Gemini EmbeddingService
- ChromaDB VectorDatabase
- DocumentIndexer
"""

import os

import pytest
from dotenv import load_dotenv

from src.chunk_models import Chunk
from src.embeddings import EmbeddingService
from src.indexing import DocumentIndexer
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
            filename="sample.txt",
            page_number=1,
            chunk_index=0,
            text="Artificial Intelligence is transforming industries.",
            start_char=0,
            end_char=53,
        ),

        Chunk(
            chunk_id="chunk_002",
            document_id="doc001",
            filename="sample.txt",
            page_number=1,
            chunk_index=1,
            text="Machine Learning is a subset of Artificial Intelligence.",
            start_char=54,
            end_char=114,
        ),
    ]


def test_single_embedding():

    embedding_service = EmbeddingService()

    embedding = embedding_service.embed_text(
        "Hello World"
    )

    assert isinstance(embedding, list)
    assert len(embedding) > 0


def test_batch_embeddings():

    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_batch(
        [
            "Hello",
            "World",
            "Artificial Intelligence",
        ]
    )

    assert len(embeddings) == 3

    for embedding in embeddings:

        assert isinstance(embedding, list)
        assert len(embedding) > 0


def test_embedding_dimension():

    embedding_service = EmbeddingService()

    dimension = embedding_service.embedding_dimension

    assert dimension > 0


def test_vector_database_insert():

    vectordb = VectorDatabase(
        collection_name="test_insert"
    )

    vectordb.reset()

    embedding_service = EmbeddingService()

    chunks = create_chunks()

    embeddings = embedding_service.embed_batch(
        [chunk.text for chunk in chunks]
    )

    stored = vectordb.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
    )

    assert stored == 2
    assert vectordb.count() == 2

    vectordb.reset()


def test_similarity_search():

    vectordb = VectorDatabase(
        collection_name="test_search"
    )

    vectordb.reset()

    embedding_service = EmbeddingService()

    chunks = create_chunks()

    embeddings = embedding_service.embed_batch(
        [chunk.text for chunk in chunks]
    )

    vectordb.add_chunks(
        chunks,
        embeddings,
    )

    query_embedding = embedding_service.embed_text(
        "What is Artificial Intelligence?"
    )

    results = vectordb.similarity_search(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(results["ids"][0]) > 0

    vectordb.reset()


def test_document_indexer():

    vectordb = VectorDatabase(
        collection_name="test_indexer"
    )

    vectordb.reset()

    indexer = DocumentIndexer(
        vector_database=vectordb
    )

    chunks = create_chunks()

    stats = indexer.index_chunks(chunks)

    assert stats["indexed_chunks"] == 2
    assert stats["total_vectors"] == 2

    indexer.reset_index()


def test_collection_information():

    vectordb = VectorDatabase(
        collection_name="test_collection_info"
    )

    info = vectordb.get_collection_info()

    assert "collection_name" in info
    assert "vector_count" in info


def test_reset_index():

    vectordb = VectorDatabase(
        collection_name="test_reset"
    )

    vectordb.reset()

    assert vectordb.count() == 0