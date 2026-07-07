"""
test_evaluation.py

Unit tests for the evaluation pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evaluation.evaluate import Evaluator


@pytest.fixture
def evaluator():
    ev = Evaluator()

    ev.retriever = MagicMock()
    ev.llm = MagicMock()

    return ev


def test_questions_file_exists():
    assert Path("evaluation/questions.json").exists()


def test_ground_truth_file_exists():
    assert Path("evaluation/ground_truth.json").exists()


def test_load_questions(evaluator):

    questions = evaluator.load_questions()

    assert isinstance(questions, list)

    if questions:
        assert "id" in questions[0]
        assert "question" in questions[0]
        assert "document_id" in questions[0]


def test_load_ground_truth(evaluator):

    ground_truth = evaluator.load_ground_truth()

    assert isinstance(ground_truth, dict)


def test_evaluate_runs(monkeypatch, evaluator):

    questions = [
        {
            "id": "q1",
            "question": "What is AI?",
            "document_id": "doc1",
        }
    ]

    ground_truth = {
        "q1": "Artificial Intelligence"
    }

    monkeypatch.setattr(
        evaluator,
        "load_questions",
        lambda: questions,
    )

    monkeypatch.setattr(
        evaluator,
        "load_ground_truth",
        lambda: ground_truth,
    )

    evaluator.retriever.retrieve_for_generation.return_value = {
        "context": "Artificial Intelligence is AI.",
        "chunks": [
            {
                "document_id": "doc1",
                "filename": "sample.pdf",
                "page_number": 1,
                "chunk_id": "c1",
                "text": "Artificial Intelligence is AI.",
                "distance": 0.05,
            }
        ],
    }

    evaluator.llm.generate_answer_with_confidence.return_value = {
        "answer": "Artificial Intelligence is AI.",
        "confidence": 95,
    }

    monkeypatch.setattr(
        "evaluation.evaluate.CitationFormatter.format_citations",
        lambda chunks: [
            {
                "filename": "sample.pdf",
                "page_number": 1,
                "chunk_id": "c1",
                "snippet": "Artificial Intelligence is AI.",
            }
        ],
    )

    evaluator.evaluate()


def test_empty_questions(monkeypatch, evaluator):

    monkeypatch.setattr(
        evaluator,
        "load_questions",
        lambda: [],
    )

    monkeypatch.setattr(
        evaluator,
        "load_ground_truth",
        lambda: {},
    )

    evaluator.evaluate()