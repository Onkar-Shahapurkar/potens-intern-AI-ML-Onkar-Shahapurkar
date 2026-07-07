"""
test_contradiction.py

Integration tests for document contradiction analysis.
"""

import os

import pytest
from dotenv import load_dotenv

from src.chunk_models import Chunk
from src.contradiction import ContradictionAnalyzer
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
            chunk_id="doc_a_chunk_001",
            document_id="doc_a",
            filename="security_policy_v1.pdf",
            page_number=1,
            chunk_index=0,
            text=(
                "Employees must change their passwords every 90 days."
            ),
            start_char=0,
            end_char=55,
        ),

        Chunk(
            chunk_id="doc_b_chunk_001",
            document_id="doc_b",
            filename="security_policy_v2.pdf",
            page_number=1,
            chunk_index=0,
            text=(
                "Passwords should only be changed after compromise "
                "or suspected compromise."
            ),
            start_char=0,
            end_char=78,
        ),

    ]


@pytest.fixture
def analyzer():

    vectordb = VectorDatabase(
        collection_name="test_contradiction"
    )

    vectordb.reset()

    indexer = DocumentIndexer(
        vector_database=vectordb
    )

    indexer.index_chunks(
        create_chunks()
    )

    retriever = Retriever(
        vector_database=vectordb
    )

    yield ContradictionAnalyzer(
        retriever=retriever,
    )

    vectordb.reset()


def test_compare_documents(analyzer):

    result = analyzer.analyze(
        topic="password policy",
        document_a_id="doc_a",
        document_b_id="doc_b",
    )

    assert isinstance(result, dict)

    assert "conflict" in result
    assert "reason" in result
    assert "evidence" in result


def test_document_ids_returned(analyzer):

    result = analyzer.analyze(
        topic="password policy",
        document_a_id="doc_a",
        document_b_id="doc_b",
    )

    assert result["document_a"] == "doc_a"
    assert result["document_b"] == "doc_b"


def test_missing_document():

    retriever = Retriever(
        vector_database=VectorDatabase(
            collection_name="missing_docs"
        )
    )

    analyzer = ContradictionAnalyzer(
        retriever=retriever,
    )

    result = analyzer.analyze(
        topic="security",
        document_a_id="unknown_a",
        document_b_id="unknown_b",
    )

    assert result["conflict"] is False
    assert "Insufficient information" in result["reason"]


def test_response_schema(analyzer):

    result = analyzer.analyze(
        topic="password policy",
        document_a_id="doc_a",
        document_b_id="doc_b",
    )

    assert isinstance(result["conflict"], bool)
    assert isinstance(result["reason"], str)
    assert isinstance(result["evidence"], list)


def test_same_document(analyzer):

    result = analyzer.analyze(
        topic="password policy",
        document_a_id="doc_a",
        document_b_id="doc_a",
    )

    assert isinstance(result, dict)


def test_empty_topic(analyzer):

    result = analyzer.analyze(
        topic="",
        document_a_id="doc_a",
        document_b_id="doc_b",
    )

    assert isinstance(result, dict)