"""
dependencies.py

Dependency providers for the FastAPI application.
"""

from __future__ import annotations

from functools import lru_cache

from src.contradiction import ContradictionAnalyzer
from src.llm import LLMService
from src.reranker import Reranker
from src.retrieval import Retriever
from src.translation import TranslationService


@lru_cache
def get_translation_service() -> TranslationService:
    """
    Return a singleton TranslationService.
    """
    return TranslationService()


lru_cache
def get_reranker():

    try:

        return Reranker()

    except Exception:

        # No HF token or reranker unavailable.
        # Retrieval will fall back to vector search only.
        return None

@lru_cache
def get_retriever():

    return Retriever(
        translator=get_translation_service(),
        reranker=get_reranker(),
    )


@lru_cache
def get_llm_service() -> LLMService:
    """
    Return a singleton LLMService.
    """
    return LLMService(
        translator=get_translation_service(),
    )


@lru_cache
def get_contradiction_analyzer() -> ContradictionAnalyzer:
    """
    Return a singleton ContradictionAnalyzer.
    """
    return ContradictionAnalyzer(
        retriever=get_retriever(),
        llm=get_llm_service(),
    )