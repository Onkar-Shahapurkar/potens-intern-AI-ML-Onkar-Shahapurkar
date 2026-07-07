"""
embeddings.py

Embedding service using Google's Gemini Embedding API.

Responsibilities:
- Load Gemini client
- Generate embeddings
- Generate batch embeddings
- Expose embedding model information
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from google import genai


load_dotenv()


class EmbeddingService:
    """
    Gemini Embedding Service
    """

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
    ):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables."
            )

        self.client = genai.Client(api_key=api_key)

        self.model_name = model_name

    def embed_text(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate embedding for a single text.
        """

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )

        return response.embeddings[0].values

    def embed_batch(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        """

        embeddings = []

        for text in texts:

            vector = self.embed_text(text)

            embeddings.append(vector)

        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """
        Determine embedding dimension dynamically.
        """

        sample = self.embed_text("Hello")

        return len(sample)

    @property
    def model_info(self) -> str:
        """
        Return model name.
        """

        return self.model_name