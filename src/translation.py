"""
translation.py

Translation service for multilingual RAG.

Responsibilities:
- Detect query language
- Translate query to English
- Translate answer back to the original language

Uses Gemini 2.5 Flash.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class TranslationService:
    """
    Translation service powered by Gemini.
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

    def detect_language(
        self,
        text: str,
    ) -> str:
        """
        Detect the language of the input text.
        Returns ISO-639-1 language code.
        """

        prompt = f"""
Identify the language of the following text.

Return ONLY the ISO-639-1 language code.

Examples:
English -> en
Hindi -> hi
Marathi -> mr
French -> fr

Text:
{text}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip().lower()

    def translate_to_english(
        self,
        text: str,
    ) -> str:
        """
        Translate text into English.
        """

        prompt = f"""
Translate the following text into English.

Return ONLY the translated text.

Text:
{text}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()

    def translate_from_english(
        self,
        text: str,
        target_language: str,
    ) -> str:
        """
        Translate English text into the target language.

        Preserve:
        - File names
        - Page numbers
        - Chunk IDs
        - Technical terms whenever appropriate.
        """

        prompt = f"""
Translate the following text into {target_language}.

Rules:
- Preserve filenames exactly.
- Preserve page numbers.
- Preserve chunk IDs.
- Preserve code blocks.
- Preserve markdown formatting.
- Return ONLY the translated text.

Text:

{text}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text.strip()

    def translate_round_trip(
        self,
        question: str,
    ) -> tuple[str, str]:
        """
        Detect language and translate to English if necessary.

        Returns:
            (language_code, english_question)
        """

        language = self.detect_language(question)

        if language == "en":
            return language, question

        english = self.translate_to_english(question)

        return language, english

    @property
    def model_info(self) -> str:
        """
        Return translation model name.
        """

        return self.model_name