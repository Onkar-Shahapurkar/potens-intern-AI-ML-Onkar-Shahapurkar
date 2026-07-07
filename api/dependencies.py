"""
dependencies.py

Dependency providers for the FastAPI application.
"""

from functools import lru_cache

from src.contradiction import ContradictionAnalyzer
from src.llm import LLMService
from src.retrieval import Retriever
from src.translation import TranslationService


@lru_cache
def get_translation_service() -> TranslationService:
    """
    Return a singleton TranslationService.
    """
    return TranslationService()


@lru_cache
def get_retriever() -> Retriever:
    """
    Return a singleton Retriever.
    """
    return Retriever()


@lru_cache
def get_llm_service() -> LLMService:
    """
    Return a singleton LLMService.
    """
    return LLMService()


@lru_cache
def get_contradiction_analyzer() -> ContradictionAnalyzer:
    """
    Return a singleton ContradictionAnalyzer.
    """
    return ContradictionAnalyzer(
        retriever=get_retriever(),
        llm=get_llm_service(),
    )