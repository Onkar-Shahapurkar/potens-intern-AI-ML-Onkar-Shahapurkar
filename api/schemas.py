"""
schemas.py

Pydantic request and response models
for the POTENS AI/ML RAG API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


# ==========================================================
# Citation
# ==========================================================

class Citation(BaseModel):
    """
    Citation returned with every answer.
    """

    filename: str
    page_number: int
    chunk_id: str
    snippet: str


# ==========================================================
# /ask
# ==========================================================

class AskRequest(BaseModel):
    """
    Request body for /ask.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask."
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of retrieved chunks."
    )


class AskResponse(BaseModel):
    """
    Response body for /ask.
    """

    answer: str

    citations: List[Citation]

    language: str


# ==========================================================
# /contradict
# ==========================================================

class Evidence(BaseModel):
    """
    Evidence returned by contradiction analysis.
    """

    document: str

    snippet: str


class ContradictRequest(BaseModel):
    """
    Request body for /contradict.
    """

    topic: str = Field(
        ...,
        min_length=1,
    )

    document_a_id: str

    document_b_id: str

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


class ContradictResponse(BaseModel):
    """
    Response body for /contradict.
    """

    conflict: bool

    reason: str

    evidence: List[Evidence]

    document_a: str

    document_b: str


# ==========================================================
# /health
# ==========================================================

class HealthResponse(BaseModel):
    """
    Response body for /health.
    """

    status: str