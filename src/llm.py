"""
llm.py

Gemini LLM service for RAG question answering.

Responsibilities:
- Call Gemini 2.5 Flash
- Generate answers using retrieved context
- Never answer outside the provided context
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from src.prompts import build_rag_prompt

load_dotenv()


class LLMService:
    """
    Wrapper around Gemini 2.5 Flash.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
    ):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model_name = model_name

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate an answer using only the supplied context.
        """

        if not context.strip():

            return (
                "I couldn't find enough information in the "
                "uploaded documents to answer this question."
            )

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()

    @property
    def model_info(self) -> str:
        """
        Return model name.
        """

        return self.model_name