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
from src.translation import TranslationService


@st.cache_resource
def get_ingestor() -> DocumentIngestor:
    """
    Return a shared DocumentIngestor instance.
    """
    return DocumentIngestor()


@st.cache_resource
def get_chunker() -> DocumentChunker:
    """
    Return a shared DocumentChunker instance.
    """
    return DocumentChunker()


@st.cache_resource
def get_indexer() -> DocumentIndexer:
    """
    Return a shared DocumentIndexer instance.
    """
    return DocumentIndexer()


@st.cache_resource
def get_translation_service() -> TranslationService:
    """
    Return a shared TranslationService instance.
    """
    return TranslationService()


@st.cache_resource
def get_retriever() -> Retriever:
    """
    Return a shared Retriever instance.
    """
    return Retriever(
        translator=get_translation_service(),
    )


@st.cache_resource
def get_llm() -> LLMService:
    """
    Return a shared LLMService instance.
    """
    return LLMService(
        translator=get_translation_service(),
    )


@st.cache_resource
def get_contradiction() -> ContradictionAnalyzer:
    """
    Return a shared ContradictionAnalyzer instance.
    """
    return ContradictionAnalyzer(
        retriever=get_retriever(),
        llm=get_llm(),
    )


def initialize_session_state() -> None:
    """
    Initialize shared Streamlit session state.
    """

    defaults = {
        "knowledge_base_ready": False,
        "indexed_documents": [],
        "uploaded_document_ids": set(),
        "index_stats": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_session_state() -> None:
    """
    Reset all session state related to the knowledge base.
    """

    st.session_state["knowledge_base_ready"] = False
    st.session_state["indexed_documents"] = []
    st.session_state["uploaded_document_ids"] = set()
    st.session_state["index_stats"] = {}

    # Clear the persisted vector collection
    get_indexer().reset_index()