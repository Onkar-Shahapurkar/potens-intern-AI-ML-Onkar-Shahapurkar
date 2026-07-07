"""
confidence.py

Confidence scoring for the RAG pipeline.

Factors considered:
- Average retrieval similarity
- Number of retrieved chunks
- Context coverage
- Citation coverage

Returns:
- Confidence score (0–100)
- Confidence level
- Human review recommendation
"""

from __future__ import annotations

from typing import Dict, List


class ConfidenceScorer:
    """
    Production-oriented confidence scorer.
    """

    HIGH_THRESHOLD = 90
    MEDIUM_THRESHOLD = 70

    def calculate(
        self,
        retrieved_chunks: List[dict],
        context: str = "",
        citations: List[dict] | None = None,
    ) -> Dict:
        """
        Calculate overall confidence.

        Parameters
        ----------
        retrieved_chunks
            Retrieved chunks from vector search.

        context
            Combined context passed to the LLM.

        citations
            Generated citations.

        Returns
        -------
        dict
        """

        citations = citations or []

        if not retrieved_chunks:

            return {
                "confidence": 0,
                "confidence_level": "Low",
                "human_review": True,
                "metrics": {
                    "retrieval": 0,
                    "coverage": 0,
                    "citations": 0,
                },
            }

        retrieval_score = self._retrieval_score(
            retrieved_chunks
        )

        coverage_score = self._coverage_score(
            context
        )

        citation_score = self._citation_score(
            citations,
            retrieved_chunks,
        )

        chunk_score = self._chunk_score(
            retrieved_chunks
        )

        if not context and not citations:

            confidence = retrieval_score

        else:

            confidence = int(
                (
                        retrieval_score * 0.50
                        + coverage_score * 0.20
                        + citation_score * 0.15
                        + chunk_score * 0.15
                )
            )

        confidence = max(
            0,
            min(
                100,
                confidence,
            ),
        )

        return {
            "confidence": confidence,
            "confidence_level": self._confidence_level(
                confidence
            ),
            "human_review": confidence
            < self.MEDIUM_THRESHOLD,
            "metrics": {
                "retrieval": retrieval_score,
                "coverage": coverage_score,
                "citations": citation_score,
                "chunks": chunk_score,
            },
        }

    def _retrieval_score(
        self,
        chunks: List[dict],
    ) -> int:
        """
        Score based on average vector distance.
        """

        distances = [
            chunk["distance"]
            for chunk in chunks
            if "distance" in chunk
               and chunk["distance"] is not None
        ]

        if not distances:
            return 0

        avg_distance = sum(distances) / len(distances)

        score = int((1 - avg_distance) * 100)

        return max(
            0,
            min(
                100,
                score,
            ),
        )

    def _coverage_score(
        self,
        context: str,
    ) -> int:
        """
        Score based on context size.
        """

        if not context:
            return 0

        length = len(context)

        if length >= 3000:
            return 100

        if length >= 2000:
            return 90

        if length >= 1000:
            return 80

        if length >= 500:
            return 70

        if length >= 250:
            return 50

        return 25

    def _citation_score(
        self,
        citations: List[dict],
        chunks: List[dict],
    ) -> int:
        """
        Score based on citation coverage.
        """

        if not chunks:
            return 0

        ratio = len(citations) / len(chunks)

        ratio = min(
            1.0,
            ratio,
        )

        return int(ratio * 100)

    def _chunk_score(
        self,
        chunks: List[dict],
    ) -> int:
        """
        Score based on number of retrieved chunks.
        """

        count = len(chunks)

        if count >= 5:
            return 100

        if count == 4:
            return 90

        if count == 3:
            return 80

        if count == 2:
            return 65

        if count == 1:
            return 40

        return 0

    def _confidence_level(
        self,
        confidence: int,
    ) -> str:

        if confidence >= self.HIGH_THRESHOLD:
            return "High"

        if confidence >= self.MEDIUM_THRESHOLD:
            return "Medium"

        return "Low"

    def review_message(
        self,
        confidence: int,
    ) -> str:
        """
        Human-in-the-loop recommendation.
        """

        if confidence >= self.HIGH_THRESHOLD:

            return (
                "🟢 High confidence. The retrieved evidence strongly "
                "supports the generated answer."
            )

        if confidence >= self.MEDIUM_THRESHOLD:

            return (
                "🟡 Medium confidence. The answer is reasonably "
                "supported, but review the cited sources if the "
                "decision is important."
            )

        return (
            "🔴 Low confidence. The retrieved evidence may not be "
            "sufficient. Please verify the cited sources before "
            "relying on this answer."
        )