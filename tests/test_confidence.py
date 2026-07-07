"""
test_confidence.py

Unit tests for confidence scoring.
"""

import pytest

from src.confidence import ConfidenceScorer


@pytest.fixture
def scorer():

    return ConfidenceScorer()


def test_high_confidence(scorer):

    chunks = [
        {"distance": 0.03},
        {"distance": 0.06},
        {"distance": 0.08},
    ]

    result = scorer.calculate(chunks)

    assert result["confidence"] >= 90
    assert result["confidence_level"] == "High"
    assert result["human_review"] is False


def test_medium_confidence(scorer):

    chunks = [
        {"distance": 0.20},
        {"distance": 0.25},
        {"distance": 0.30},
    ]

    result = scorer.calculate(chunks)

    assert 70 <= result["confidence"] < 90
    assert result["confidence_level"] == "Medium"
    assert result["human_review"] is False


def test_low_confidence(scorer):

    chunks = [
        {"distance": 0.65},
        {"distance": 0.70},
        {"distance": 0.75},
    ]

    result = scorer.calculate(chunks)

    assert result["confidence"] < 70
    assert result["confidence_level"] == "Low"
    assert result["human_review"] is True


def test_empty_chunks(scorer):

    result = scorer.calculate([])

    assert result["confidence"] == 0
    assert result["confidence_level"] == "Low"
    assert result["human_review"] is True


def test_missing_distance(scorer):

    chunks = [
        {},
        {},
    ]

    result = scorer.calculate(chunks)

    assert result["confidence"] == 0
    assert result["confidence_level"] == "Low"
    assert result["human_review"] is True


def test_confidence_level_high(scorer):

    assert (
        scorer._confidence_level(95)
        == "High"
    )


def test_confidence_level_medium(scorer):

    assert (
        scorer._confidence_level(80)
        == "Medium"
    )


def test_confidence_level_low(scorer):

    assert (
        scorer._confidence_level(40)
        == "Low"
    )


def test_high_review_message(scorer):

    message = scorer.review_message(95)

    assert "High confidence" in message


def test_medium_review_message(scorer):

    message = scorer.review_message(75)

    assert "Medium confidence" in message


def test_low_review_message(scorer):

    message = scorer.review_message(30)

    assert "Low confidence" in message