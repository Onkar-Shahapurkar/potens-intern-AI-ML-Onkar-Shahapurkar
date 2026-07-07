"""
chunk_models.py

Data models for document chunks.
"""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Represents a single chunk extracted from a document.
    """

    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier"
    )

    document_id: str = Field(
        ...,
        description="Parent document identifier"
    )

    filename: str = Field(
        ...,
        description="Original document filename"
    )

    page_number: int = Field(
        ...,
        ge=1,
        description="Page where the chunk originated"
    )

    chunk_index: int = Field(
        ...,
        ge=0,
        description="Sequential chunk number within the page"
    )

    text: str = Field(
        ...,
        description="Chunk content"
    )

    start_char: int = Field(
        ...,
        ge=0,
        description="Starting character offset"
    )

    end_char: int = Field(
        ...,
        ge=0,
        description="Ending character offset"
    )

    @property
    def length(self) -> int:
        """
        Length of the chunk in characters.
        """
        return len(self.text)


class ChunkCollection(BaseModel):
    """
    Collection of chunks belonging to one document.
    """

    document_id: str
    total_chunks: int
    chunks: list[Chunk]