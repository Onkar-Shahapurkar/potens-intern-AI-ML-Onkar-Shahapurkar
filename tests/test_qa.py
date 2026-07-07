"""
test_qa.py

Integration tests for the RAG Question Answering pipeline.
"""

import os

import pytest
from dotenv import load_dotenv

from src.chunk_models import Chunk
from src.citations import CitationFormatter
from src.indexing import DocumentIndexer
from src.llm import LLMService
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
            filename="rag.pdf",
            page_number=1,
            chunk_index=0,
            text=(
                "Retrieval-Augmented Generation (RAG) combines "
                "retrieval with a large language model."
            ),
            start_char=0,
            end_char=85,
        ),
        Chunk(
            chunk_id="chunk_002",
            document_id="doc001",
            filename="rag.pdf",
            page_number=2,
            chunk_index=1,
            text=(
                "ChromaDB is a vector database used to store "
                "embeddings for semantic search."
            ),
            start_char=86,
            end_char=170,
        ),
    ]


@pytest.fixture
def rag_pipeline():

    vectordb = VectorDatabase(
        collection_name="test_rag_pipeline"
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

    llm = LLMService()

    yield retriever, llm, vectordb

    vectordb.reset()


def test_answer_generation(rag_pipeline):

    retriever, llm, _ = rag_pipeline

    context = retriever.retrieve_context(
        query="What is Retrieval-Augmented Generation?",
        top_k=2,
    )

    answer = llm.generate_answer(
        question="What is Retrieval-Augmented Generation?",
        context=context,
    )

    assert isinstance(answer, str)
    assert len(answer) > 0


def test_hallucination_guard():

    llm = LLMService()

    answer = llm.generate_answer(
        question="Who won the FIFA World Cup in 2010?",
        context="",
    )

    assert (
        "couldn't find enough information"
        in answer.lower()
    )


def test_retrieved_context_exists(rag_pipeline):

    retriever, _, _ = rag_pipeline

    context = retriever.retrieve_context(
        query="vector database",
        top_k=2,
    )

    assert isinstance(context, str)
    assert len(context) > 0


def test_citation_generation(rag_pipeline):

    retriever, _, _ = rag_pipeline

    results = retriever.retrieve(
        query="ChromaDB",
        top_k=2,
    )

    citations = CitationFormatter.format_citations(
        results
    )

    assert len(citations) > 0

    citation = citations[0]

    assert "filename" in citation
    assert "page_number" in citation
    assert "chunk_id" in citation
    assert "snippet" in citation


def test_markdown_citations(rag_pipeline):

    retriever, _, _ = rag_pipeline

    results = retriever.retrieve(
        query="RAG",
        top_k=2,
    )

    citations = CitationFormatter.format_citations(
        results
    )

    markdown = CitationFormatter.format_markdown(
        citations
    )

    assert "Sources" in markdown
    assert "Chunk" in markdown


def test_json_citations(rag_pipeline):

    retriever, _, _ = rag_pipeline

    results = retriever.retrieve(
        query="semantic search",
        top_k=2,
    )

    citations = CitationFormatter.format_citations(
        results
    )

    response = CitationFormatter.format_json(
        citations
    )

    assert response["total_sources"] > 0
    assert isinstance(response["citations"], list)


def test_no_retrieval_results():

    vectordb = VectorDatabase(
        collection_name="empty_rag_collection"
    )

    vectordb.reset()

    retriever = Retriever(
        vector_database=vectordb
    )

    results = retriever.retrieve(
        query="Quantum Computing",
        top_k=5,
    )

    assert results == []

    vectordb.reset()