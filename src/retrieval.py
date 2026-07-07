"""
retrieval.py

Semantic retrieval service.

Responsibilities:
- Multilingual semantic retrieval
- Document-specific retrieval
- Context generation for LLM
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.embeddings import EmbeddingService
from src.translation import TranslationService
from src.vectordb import VectorDatabase


class Retriever:
    """
    Semantic retriever for the RAG pipeline.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_database: VectorDatabase | None = None,
        translator: TranslationService | None = None,
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

        self.translator = (
            translator
            if translator
            else TranslationService()
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from all indexed documents.
        """

        _, english_query = (
            self.translator.translate_round_trip(query)
        )

        return self._search(
            query=english_query,
            top_k=top_k,
            where=None,
        )

    def retrieve_from_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from one document.
        """

        _, english_query = (
            self.translator.translate_round_trip(query)
        )

        return self._search(
            query=english_query,
            top_k=top_k,
            where={
                "document_id": document_id,
            },
        )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Build context from all retrieved chunks.
        """

        chunks = self.retrieve(
            query=query,
            top_k=top_k,
        )

        return self._build_context(chunks)

    def retrieve_document_context(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
    ) -> str:
        """
        Build context from one document.
        """

        chunks = self.retrieve_from_document(
            query=query,
            document_id=document_id,
            top_k=top_k,
        )

        return self._build_context(chunks)

    def retrieve_for_generation(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Retrieve everything needed for answer generation.
        """

        language, english_query = (
            self.translator.translate_round_trip(query)
        )

        chunks = self._search(
            query=english_query,
            top_k=top_k,
            where=None,
        )

        return {
            "language": language,
            "translated_query": english_query,
            "context": self._build_context(chunks),
            "chunks": chunks,
        }

    def _search(
        self,
        query: str,
        top_k: int,
        where: Dict[str, Any] | None,
    ) -> List[Dict[str, Any]]:
        """
        Internal semantic search.
        """

        if not query.strip():
            return []

        query_embedding = self.embedding_service.embed_text(
            query
        )

        results = self.vector_database.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )

        return self._format_results(results)

    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Convert retrieved chunks into LLM context.
        """

        if not chunks:
            return ""

        context = []

        for chunk in chunks:

            context.append(
                "\n".join(
                    [
                        f"[Source: {chunk['filename']}]",
                        f"[Page: {chunk['page_number']}]",
                        f"[Chunk ID: {chunk['chunk_id']}]",
                        chunk["text"],
                    ]
                )
            )

        return "\n\n".join(context)

    def _format_results(
        self,
        results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Convert ChromaDB response into structured dictionaries.
        """

        formatted: List[Dict[str, Any]] = []

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

            formatted.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id"),
                    "filename": metadata.get("filename"),
                    "page_number": metadata.get("page_number"),
                    "chunk_index": metadata.get("chunk_index"),
                    "start_char": metadata.get("start_char"),
                    "end_char": metadata.get("end_char"),
                    "text": document,
                    "distance": float(distance),
                }
            )

        return formatted