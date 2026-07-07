"""
cleaner.py

Utilities for cleaning and normalizing extracted document text.

Responsibilities:
- Normalize whitespace
- Normalize line breaks
- Remove empty lines
- Preserve document meaning for accurate citations
"""

import re


def clean_text(text: str) -> str:
    """
    Clean extracted document text while preserving meaning.

    Parameters
    ----------
    text : str
        Raw extracted text.

    Returns
    -------
    str
        Cleaned text.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing spaces on each line
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Join lines back together
    text = "\n".join(lines)

    # Collapse multiple spaces into one
    text = re.sub(r"[ ]{2,}", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_empty(text: str) -> bool:
    """
    Check whether the cleaned text is effectively empty.
    """

    return len(text.strip()) == 0


def word_count(text: str) -> int:
    """
    Count the number of words in the text.
    """

    return len(text.split())


def character_count(text: str) -> int:
    """
    Count the number of characters in the text.
    """

    return len(text)


def preview(text: str, max_length: int = 200) -> str:
    """
    Generate a short preview of the document.
    """

    text = text.strip()

    if len(text) <= max_length:
        return text

    return text[:max_length].rstrip() + "..."