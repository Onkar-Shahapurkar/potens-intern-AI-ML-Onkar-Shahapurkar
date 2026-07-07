"""
test_translation.py

Integration tests for multilingual translation support.
"""

import os

import pytest
from dotenv import load_dotenv

from src.translation import TranslationService

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

pytestmark = pytest.mark.skipif(
    not API_KEY,
    reason="GEMINI_API_KEY not configured."
)


@pytest.fixture
def translator():
    return TranslationService()


def test_detect_english(translator):

    language = translator.detect_language(
        "What is Retrieval Augmented Generation?"
    )

    assert language == "en"


def test_detect_hindi(translator):

    language = translator.detect_language(
        "रिट्रीवल ऑगमेंटेड जनरेशन क्या है?"
    )

    assert language == "hi"


def test_detect_marathi(translator):

    language = translator.detect_language(
        "रिट्रिव्हल ऑगमेंटेड जनरेशन म्हणजे काय?"
    )

    assert language == "mr"


def test_translate_hindi_to_english(translator):

    english = translator.translate_to_english(
        "रिट्रीवल ऑगमेंटेड जनरेशन क्या है?"
    )

    assert isinstance(english, str)
    assert len(english) > 0


def test_translate_marathi_to_english(translator):

    english = translator.translate_to_english(
        "रिट्रिव्हल ऑगमेंटेड जनरेशन म्हणजे काय?"
    )

    assert isinstance(english, str)
    assert len(english) > 0


def test_translate_from_english(translator):

    translated = translator.translate_from_english(
        "Retrieval Augmented Generation combines retrieval and LLMs.",
        "Hindi",
    )

    assert isinstance(translated, str)
    assert len(translated) > 0


def test_round_trip_english(translator):

    language, query = translator.translate_round_trip(
        "What is ChromaDB?"
    )

    assert language == "en"
    assert query == "What is ChromaDB?"


def test_round_trip_hindi(translator):

    language, query = translator.translate_round_trip(
        "क्रोमा डीबी क्या है?"
    )

    assert language == "hi"
    assert isinstance(query, str)
    assert len(query) > 0


def test_round_trip_marathi(translator):

    language, query = translator.translate_round_trip(
        "क्रोमा डीबी म्हणजे काय?"
    )

    assert language == "mr"
    assert isinstance(query, str)
    assert len(query) > 0


def test_empty_string(translator):

    language = translator.detect_language("")

    assert isinstance(language, str)


def test_model_info(translator):

    assert translator.model_info == "gemini-2.5-flash"