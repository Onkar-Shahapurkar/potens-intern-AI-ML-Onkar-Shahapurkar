"""
test_api.py

Integration tests for the FastAPI endpoints.
"""

import os

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == "POTENS AI/ML RAG API"

    assert data["status"] == "running"


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_empty_question():

    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5,
        },
    )

    assert response.status_code in (400, 422)


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not configured.",
)
def test_ask():

    response = client.post(
        "/ask",
        json={
            "question": "What is Retrieval-Augmented Generation?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data
    assert "language" in data


def test_invalid_contradiction_request():

    response = client.post(
        "/contradict",
        json={
            "topic": "",
            "document_a_id": "",
            "document_b_id": "",
            "top_k": 5,
        },
    )

    assert response.status_code in (400, 422)


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY not configured.",
)
def test_contradict():

    response = client.post(
        "/contradict",
        json={
            "topic": "Password Policy",
            "document_a_id": "doc_a",
            "document_b_id": "doc_b",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "conflict" in data
    assert "reason" in data
    assert "evidence" in data
    assert "document_a" in data
    assert "document_b" in data


def test_invalid_endpoint():

    response = client.get("/does-not-exist")

    assert response.status_code == 404