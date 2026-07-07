"""
indexing.py

Coordinates the indexing pipeline.

Pipeline:

Document
    ↓
Chunking
    ↓
Gemini Embeddings
    ↓
ChromaDB
"""

from __future__ import annotations

from typing import List, Dict

from src.chunk_models import Chunk
from src.embeddings import EmbeddingService
from src.vectordb import VectorDatabase


class DocumentIndexer:
    """
    Coordinates embedding generation and vector indexing.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_database: VectorDatabase | None = None,
    ):

        self.embedding_service = (
            embedding_service
            if embedding_service
            else EmbeddingService()
        )

        self.vector_database = (
            vector_database
            if vector_database
            else VectorDatabase()
        )

    def index_chunks(
        self,
        chunks: List[Chunk],
    ) -> Dict:
        """
        Generate embeddings and store them.

        Returns indexing statistics.
        """

        if not chunks:

            return {
                "indexed_chunks": 0,
                "embedding_model": self.embedding_service.model_info,
                "collection_name": self.vector_database.collection_name,
                "total_vectors": self.vector_database.count(),
            }

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_service.embed_batch(
            texts
        )

        stored = self.vector_database.add_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        return {
            "indexed_chunks": stored,
            "embedding_model": self.embedding_service.model_info,
            "collection_name": self.vector_database.collection_name,
            "total_vectors": self.vector_database.count(),
        }

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
    ):
        """
        Search indexed chunks.
        """

        query_embedding = self.embedding_service.embed_text(
            query
        )

        return self.vector_database.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

    def indexed_vectors(self) -> int:
        """
        Return total indexed vectors.
        """

        return self.vector_database.count()

    def collection_info(self):
        """
        Return collection information.
        """

        return self.vector_database.get_collection_info()

    def reset_index(self):
        """
        Delete all indexed vectors.
        """

        self.vector_database.reset()