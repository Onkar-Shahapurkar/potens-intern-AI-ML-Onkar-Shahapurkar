"""
retrieval.py

Semantic retrieval with reranking support.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.embeddings import EmbeddingService
from src.reranker import Reranker
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
        reranker: Reranker | None = None,
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

        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks from all documents.
        """

        _, english_query = (
            self.translator.translate_round_trip(query)
        )

        return self._search(
            query=english_query,
            top_k=top_k,
            where=None,
            rerank=rerank,
        )

    def retrieve_from_document(
        self,
        query: str,
        document_id: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from a specific document.
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
            rerank=rerank,
        )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:

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
        initial_k: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve and rerank chunks for answer generation.
        """

        language, english_query = (
            self.translator.translate_round_trip(query)
        )

        chunks = self._search(
            query=english_query,
            top_k=top_k,
            initial_k=initial_k,
            where=None,
            rerank=True,
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
        rerank: bool = True,
        initial_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Internal semantic search.
        """

        if not query.strip():
            return []

        query_embedding = self.embedding_service.embed_text(
            query
        )

        search_k = initial_k or top_k

        results = self.vector_database.similarity_search(
            query_embedding=query_embedding,
            top_k=search_k,
            where=where,
        )

        chunks = self._format_results(results)

        if (
            rerank
            and self.reranker is not None
            and chunks
        ):

            chunks = self.reranker.rerank(
                query=query,
                chunks=chunks,
                top_k=top_k,
            )

        else:

            chunks = chunks[:top_k]

        return chunks

    def _build_context(
        self,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build LLM context.
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
        Convert ChromaDB response into dictionaries.
        """

        formatted = []

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