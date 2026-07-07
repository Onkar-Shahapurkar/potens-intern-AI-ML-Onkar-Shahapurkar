"""
api/routes.py

FastAPI routes for the POTENS AI/ML RAG system.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import (
    get_contradiction_analyzer,
    get_llm_service,
    get_retriever,
    get_translation_service,
)
from api.schemas import (
    AskRequest,
    AskResponse,
    ContradictRequest,
    ContradictResponse,
    HealthResponse,
)
from src.citations import CitationFormatter
from src.contradiction import ContradictionAnalyzer
from src.llm import LLMService
from src.retrieval import Retriever
from src.translation import TranslationService

router = APIRouter()


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["Question Answering"],
)
def ask(
    request: AskRequest,
    retriever: Retriever = Depends(get_retriever),
    llm: LLMService = Depends(get_llm_service),
    translator: TranslationService = Depends(get_translation_service),
):
    """
    Answer a question using the indexed documents.
    """

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retrieval = retriever.retrieve_for_generation(
        query=question,
        top_k=request.top_k,
    )

    answer = llm.generate_answer(
        question=question,
        context=retrieval["context"],
    )

    citations = CitationFormatter.format_citations(
        retrieval["chunks"]
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        language=retrieval["language"],
    )


@router.post(
    "/contradict",
    response_model=ContradictResponse,
    tags=["Contradiction Analysis"],
)
def contradict(
    request: ContradictRequest,
    analyzer: ContradictionAnalyzer = Depends(
        get_contradiction_analyzer
    ),
):
    """
    Compare two indexed documents.
    """

    if not request.topic.strip():
        raise HTTPException(
            status_code=400,
            detail="Topic cannot be empty.",
        )

    result = analyzer.analyze(
        topic=request.topic,
        document_a_id=request.document_a_id,
        document_b_id=request.document_b_id,
        top_k=request.top_k,
    )

    return ContradictResponse(**result)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health():
    """
    Health check endpoint.
    """

    return HealthResponse(
        status="ok",
    )