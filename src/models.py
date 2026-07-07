"""
models.py

Shared data models for the document ingestion pipeline.
"""

from typing import List

from pydantic import BaseModel, Field


class Page(BaseModel):
    """
    Represents a single page of a document.
    """

    page_number: int = Field(..., ge=1)
    text: str


class DocumentMetadata(BaseModel):
    """
    Metadata associated with a document.
    """

    document_id: str
    filename: str
    file_type: str
    page_count: int
    character_count: int
    word_count: int
    language: str
    uploaded_at: str


class Document(BaseModel):
    """
    Standardized document object returned after ingestion.
    """

    metadata: DocumentMetadata
    pages: List[Page]


class IngestionResponse(BaseModel):
    """
    Response returned after successful ingestion.
    """

    success: bool = True
    message: str
    document: Document