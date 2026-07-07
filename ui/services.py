"""
ui/services.py

Shared cached services for the Streamlit application.
"""

from __future__ import annotations

import streamlit as st

from src.chunking import DocumentChunker
from src.contradiction import ContradictionAnalyzer
from src.indexing import DocumentIndexer
from src.ingestion import DocumentIngestor
from src.llm import LLMService
from src.retrieval import Retriever


@st.cache_resource
def get_ingestor() -> DocumentIngestor:
    return DocumentIngestor()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    return DocumentChunker()


@st.cache_resource
def get_indexer() -> DocumentIndexer:
    return DocumentIndexer()


@st.cache_resource
def get_retriever() -> Retriever:
    return Retriever()


@st.cache_resource
def get_llm() -> LLMService:
    return LLMService()


@st.cache_resource
def get_contradiction() -> ContradictionAnalyzer:
    return ContradictionAnalyzer(
        retriever=get_retriever(),
        llm=get_llm(),
    )