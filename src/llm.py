"""
llm.py

Gemini LLM service.

Responsibilities:
- RAG Question Answering
- Contradiction Analysis
- Multilingual Answer Generation
- Generic Prompt Execution
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

from src.prompts import build_rag_prompt
from src.translation import TranslationService

load_dotenv()


class LLMService:
    """
    Wrapper around Gemini 2.5 Flash.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        translator: TranslationService | None = None,
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

        self.translator = (
            translator
            if translator
            else TranslationService(
                model_name=model_name
            )
        )

    def generate_raw(
        self,
        prompt: str,
    ) -> str:
        """
        Execute any prompt against Gemini.
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
        Generate a grounded answer.

        Workflow:
        1. Detect query language.
        2. Translate to English (if needed).
        3. Generate answer from retrieved context.
        4. Translate answer back to original language.
        """

        if not context.strip():

            return (
                "I couldn't find enough information in the "
                "uploaded documents to answer this question."
            )

        language, english_question = (
            self.translator.translate_round_trip(
                question
            )
        )

        prompt = build_rag_prompt(
            question=english_question,
            context=context,
        )

        english_answer = self.generate_raw(
            prompt
        )

        if language == "en":

            return english_answer

        return self.translator.translate_from_english(
            english_answer,
            language,
        )

    def generate_contradiction_analysis(
        self,
        prompt: str,
    ) -> str:
        """
        Execute contradiction analysis prompt.
        Returns raw JSON string.
        """

        return self.generate_raw(prompt)

    def generate_translation(
        self,
        text: str,
        target_language: str,
    ) -> str:
        """
        Translate text into the target language.
        """

        return self.translator.translate_from_english(
            text=text,
            target_language=target_language,
        )

    @property
    def model_info(self) -> str:
        """
        Return model name.
        """

        return self.model_name