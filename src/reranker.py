"""
reranker.py

Online reranker using Hugging Face Inference API.

Model:
    cross-encoder/ms-marco-MiniLM-L6-v2

Responsibilities:
- Rerank retrieved chunks
- Return the best Top-K chunks

Environment Variables
---------------------
HF_API_TOKEN=your_huggingface_token
"""

from __future__ import annotations

import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()


class Reranker:
    """
    Online reranker using Hugging Face Inference API.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):

        self.model_name = model_name

        self.api_token = os.getenv("HF_API_TOKEN")

        self.enabled = bool(self.api_token)


        self.url = (
            f"https://api-inference.huggingface.co/models/{model_name}"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_token}"
        }

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank retrieved chunks.

        Parameters
        ----------
        query
            User question.

        chunks
            Retrieved chunks from vector search.

        top_k
            Number of chunks to return.
        """

        if not chunks:
            return []

        scored_chunks = []

        for chunk in chunks:

            score = self._score(
                query=query,
                document=chunk["text"],
            )

            chunk["rerank_score"] = score

            scored_chunks.append(chunk)

        scored_chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True,
        )

        return scored_chunks[:top_k]

    def _score(
        self,
        query: str,
        document: str,
    ) -> float:
        """
        Compute relevance score using Hugging Face API.
        """

        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": [
                    document,
                ],
            }
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        result = response.json()

        if (
            isinstance(result, list)
            and len(result) > 0
        ):
            return float(result[0])

        return 0.0

    @property
    def model_info(self) -> str:
        """
        Return reranker model name.
        """

        return self.model_name