"""
ui/api_client.py

HTTP client for communicating with the FastAPI backend.
"""

from __future__ import annotations

import os
from typing import Any

import requests


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000",
)


class APIClient:
    """
    Client for interacting with the RAG API.
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout: int = 120,
    ):

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        """
        GET /health
        """

        response = requests.get(
            f"{self.base_url}/health",
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def ask(
        self,
        question: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        POST /ask
        """

        payload = {
            "question": question,
            "top_k": top_k,
        }

        response = requests.post(
            f"{self.base_url}/ask",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def contradict(
        self,
        topic: str,
        document_a_id: str,
        document_b_id: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """
        POST /contradict
        """

        payload = {
            "topic": topic,
            "document_a_id": document_a_id,
            "document_b_id": document_b_id,
            "top_k": top_k,
        }

        response = requests.post(
            f"{self.base_url}/contradict",
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def is_available(self) -> bool:
        """
        Check whether the API is reachable.
        """

        try:

            self.health()

            return True

        except Exception:

            return False


_client = APIClient()


def get_api_client() -> APIClient:
    """
    Return a shared API client instance.
    """

    return _client