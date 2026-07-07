"""
llm.py

Gemini LLM service.

Responsibilities:
- RAG Question Answering
- Contradiction Analysis
- Generic Prompt Execution
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

    def generate_raw(
        self,
        prompt: str,
    ) -> str:
        """
        Execute any prompt against Gemini and
        return the raw response text.
        """

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()

    def generate_answer(
        self,
        question: str,
        context: str,
    ) -> str:
        """
        Generate a grounded answer using only the
        retrieved document context.
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

        return self.generate_raw(prompt)

    def generate_contradiction_analysis(
        self,
        prompt: str,
    ) -> str:
        """
        Execute contradiction analysis prompt.

        Returns raw JSON text from Gemini.
        """

        return self.generate_raw(prompt)

    @property
    def model_info(self) -> str:
        """
        Return the model name.
        """

        return self.model_name