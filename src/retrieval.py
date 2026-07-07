"""
retrieval.py

Semantic retrieval service.

Responsibilities:
- Generate query embeddings
- Search ChromaDB
- Convert search results into structured objects
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.embeddings import EmbeddingService
from src.vectordb import VectorDatabase


class Retriever:
    """
    Retrieves the most relevant document chunks
    from the vector database.
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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks.
        """

        if not query.strip():
            return []

        query_embedding = self.embedding_service.embed_text(query)

        results = self.vector_database.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return self._format_results(results)

    def _format_results(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert ChromaDB response into a cleaner format.
        """

        formatted_results: List[Dict[str, Any]] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            formatted_results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "start_char": metadata.get("start_char"),
                    "end_char": metadata.get("end_char"),
                    "text": document,
                    "distance": distance,
                }
            )

        return formatted_results

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve the top-k chunks and combine them
        into a single context string.
        """

        retrieved_chunks = self.retrieve(
            query=query,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return ""

        context = []

        for chunk in retrieved_chunks:

            context.append(
                f"[Source: {chunk['filename']} | "
                f"Page {chunk['page_number']}]\n"
                f"{chunk['text']}"
            )

        return "\n\n".join(context)