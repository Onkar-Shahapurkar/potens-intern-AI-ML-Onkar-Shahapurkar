"""
routes.py

API routes for the POTENS AI/ML RAG system.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
    AskRequest,
    AskResponse,
    ContradictRequest,
    ContradictResponse,
)
from src.citations import CitationFormatter
from src.contradiction import ContradictionAnalyzer
from src.llm import LLMService
from src.retrieval import Retriever
from src.translation import TranslationService

router = APIRouter()

from fastapi import Depends

from api.dependencies import (
    get_contradiction_analyzer,
    get_llm_service,
    get_retriever,
    get_translation_service,
)


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["Question Answering"],
)
def ask(
    request: AskRequest,
):
    """
    Answer a question using the indexed documents.
    """

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    retrieval = retriever.retrieve_for_generation(
        query=request.question,
        top_k=request.top_k,
    )

    answer = llm.generate_answer(
        question=request.question,
        context=retrieval["context"],
    )

    citations = CitationFormatter.format_citations(
        retrieval["chunks"]
    )

    language = translator.detect_language(
        request.question
    )

    return AskResponse(
        answer=answer,
        citations=citations,
        language=language,
    )


@router.post(
    "/contradict",
    response_model=ContradictResponse,
    tags=["Contradiction Analysis"],
)
def contradict(
    request: ContradictRequest,
):
    """
    Compare two indexed documents.
    """

    result = contradiction.analyze(
        topic=request.topic,
        document_a_id=request.document_a_id,
        document_b_id=request.document_b_id,
        top_k=request.top_k,
    )

    return ContradictResponse(**result)


@router.get(
    "/health",
    tags=["System"],
)
def health():
    """
    Health check.
    """

    return {
        "status": "ok",
    }