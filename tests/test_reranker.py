"""
test_reranker.py

Unit tests for the Reranker.
"""

from unittest.mock import patch

from src.reranker import Reranker


def test_empty_chunks():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        results = reranker.rerank(
            query="What is AI?",
            chunks=[],
        )

        assert results == []


def test_single_chunk():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        chunks = [
            {
                "chunk_id": "1",
                "text": "Artificial Intelligence is the simulation of human intelligence.",
            }
        ]

        with patch.object(
            reranker,
            "_score",
            return_value=0.95,
        ):

            results = reranker.rerank(
                query="What is AI?",
                chunks=chunks,
            )

            assert len(results) == 1
            assert (
                results[0]["rerank_score"]
                == 0.95
            )


def test_ranking_order():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        chunks = [
            {
                "chunk_id": "1",
                "text": "Document A",
            },
            {
                "chunk_id": "2",
                "text": "Document B",
            },
            {
                "chunk_id": "3",
                "text": "Document C",
            },
        ]

        scores = [0.35, 0.97, 0.62]

        with patch.object(
            reranker,
            "_score",
            side_effect=scores,
        ):

            results = reranker.rerank(
                query="AI",
                chunks=chunks,
            )

            assert (
                results[0]["chunk_id"]
                == "2"
            )

            assert (
                results[1]["chunk_id"]
                == "3"
            )

            assert (
                results[2]["chunk_id"]
                == "1"
            )


def test_top_k():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        chunks = [
            {
                "chunk_id": str(i),
                "text": f"Chunk {i}",
            }
            for i in range(10)
        ]

        with patch.object(
            reranker,
            "_score",
            side_effect=[
                0.10,
                0.20,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
            ],
        ):

            results = reranker.rerank(
                query="AI",
                chunks=chunks,
                top_k=5,
            )

            assert len(results) == 5

            assert (
                results[0]["chunk_id"]
                == "9"
            )

            assert (
                results[-1]["chunk_id"]
                == "5"
            )


def test_duplicate_scores():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        chunks = [
            {
                "chunk_id": "1",
                "text": "A",
            },
            {
                "chunk_id": "2",
                "text": "B",
            },
        ]

        with patch.object(
            reranker,
            "_score",
            return_value=0.80,
        ):

            results = reranker.rerank(
                query="AI",
                chunks=chunks,
            )

            assert len(results) == 2

            assert (
                results[0]["rerank_score"]
                == results[1]["rerank_score"]
            )


def test_model_info():

    with patch.dict(
        "os.environ",
        {"HF_API_TOKEN": "dummy_token"},
    ):

        reranker = Reranker()

        assert (
            reranker.model_info
            == "cross-encoder/ms-marco-MiniLM-L6-v2"
        )