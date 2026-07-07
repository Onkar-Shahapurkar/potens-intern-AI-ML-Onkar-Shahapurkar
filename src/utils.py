"""
utils.py

General utility functions used across the document ingestion pipeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def get_file_extension(filename: str) -> str:
    """
    Return the lowercase file extension.

    Example:
        report.PDF -> .pdf
    """
    return Path(filename).suffix.lower()


def is_supported_file(filename: str) -> bool:
    """
    Check whether the uploaded file type is supported.
    """
    return get_file_extension(filename) in SUPPORTED_EXTENSIONS


def calculate_file_hash(file: BinaryIO) -> str:
    """
    Calculate SHA-256 hash of a file.

    Useful for duplicate detection.
    """

    current_position = file.tell()

    file.seek(0)

    hasher = hashlib.sha256()

    while True:

        chunk = file.read(8192)

        if not chunk:
            break

        hasher.update(chunk)

    file.seek(current_position)

    return hasher.hexdigest()


def get_file_size(file: BinaryIO) -> int:
    """
    Return file size in bytes.
    """

    current_position = file.tell()

    file.seek(0, 2)

    size = file.tell()

    file.seek(current_position)

    return size


def validate_file_size(
    file: BinaryIO,
    max_size_mb: int = 20,
) -> None:
    """
    Raise an exception if file exceeds maximum size.
    """

    size = get_file_size(file)

    max_bytes = max_size_mb * 1024 * 1024

    if size > max_bytes:
        raise ValueError(
            f"File exceeds {max_size_mb} MB limit."
        )


def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe characters from filenames.
    """

    invalid_chars = '<>:"/\\|?*'

    cleaned = filename

    for char in invalid_chars:
        cleaned = cleaned.replace(char, "_")

    return cleaned.strip()


def ensure_not_empty(filename: str) -> None:
    """
    Validate filename is not empty.
    """

    if not filename.strip():
        raise ValueError("Filename cannot be empty.")