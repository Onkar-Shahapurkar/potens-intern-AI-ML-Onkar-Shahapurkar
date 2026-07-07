"""
vectordb.py

Vector database service using ChromaDB.

Responsibilities:
- Create / load collections
- Store chunk embeddings
- Perform similarity search
- Manage collections
"""

from __future__ import annotations

from typing import List, Dict, Any

import chromadb

from src.chunk_models import Chunk


class VectorDatabase:
    """
    Wrapper around ChromaDB.
    """

    def __init__(
        self,
        collection_name: str = "potens_rag",
        persist_directory: str = "./chroma_db",
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "POTENS AI/ML RAG Collection"
            }
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Store chunks and embeddings.

        Returns
        -------
        int
            Number of stored chunks.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk.chunk_id)

            documents.append(chunk.text)

            metadatas.append(
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Perform vector similarity search.
        """

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        """
        Return the total number of indexed chunks.
        """

        return self.collection.count()

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Return collection information.
        """

        return {
            "collection_name": self.collection.name,
            "vector_count": self.count(),
        }

    def delete_chunk(
        self,
        chunk_id: str,
    ) -> None:
        """
        Delete a chunk from the collection.
        """

        self.collection.delete(
            ids=[chunk_id]
        )

    def reset(self) -> None:
        """
        Delete all vectors from the collection.
        """

        collection_name = self.collection.name

        self.client.delete_collection(
            collection_name
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    @property
    def collection_name(self) -> str:
        """
        Return collection name.
        """

        return self.collection.name