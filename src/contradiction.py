"""
contradiction.py

Document contradiction analysis service.

Responsibilities:
- Retrieve relevant context from two specific documents
- Compare them using Gemini
- Return structured JSON
"""

from __future__ import annotations

import json
from typing import Dict

from src.llm import LLMService
from src.prompts import build_contradiction_prompt
from src.retrieval import Retriever


class ContradictionAnalyzer:
    """
    Compare two indexed documents on a specific topic.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: LLMService | None = None,
    ):

        self.retriever = (
            retriever
            if retriever
            else Retriever()
        )

        self.llm = (
            llm
            if llm
            else LLMService()
        )

    def analyze(
        self,
        topic: str,
        document_a_id: str,
        document_b_id: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Compare two documents and detect contradictions.
        """

        context_a = self.retriever.retrieve_document_context(
            query=topic,
            document_id=document_a_id,
            top_k=top_k,
        )

        context_b = self.retriever.retrieve_document_context(
            query=topic,
            document_id=document_b_id,
            top_k=top_k,
        )

        if not context_a or not context_b:

            return {
                "conflict": False,
                "reason": (
                    "Insufficient information available to compare "
                    "the selected documents."
                ),
                "evidence": [],
            }

        prompt = build_contradiction_prompt(
            topic=topic,
            document_a=context_a,
            document_b=context_b,
        )

        response = self.llm.generate_contradiction_analysis(
            prompt
        )

        try:

            result = json.loads(response)

        except json.JSONDecodeError:

            result = {
                "conflict": False,
                "reason": (
                    "The model returned an invalid response."
                ),
                "evidence": [],
            }

        result["document_a"] = document_a_id
        result["document_b"] = document_b_id

        return result