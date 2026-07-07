"""
parsers.py

Handles text extraction from supported document formats.

Supported:
- PDF
- DOCX
- TXT

Returns:
[
    {
        "page_number": int,
        "text": str
    }
]
"""

from io import BytesIO
from typing import BinaryIO, List, Dict

import fitz  # PyMuPDF
from docx import Document as DocxDocument


class DocumentParserError(Exception):
    """Raised when document parsing fails."""


def parse_document(
    file: BinaryIO,
    extension: str,
) -> List[Dict]:
    """
    Dispatch to the correct parser based on file extension.
    """

    extension = extension.lower()

    if extension == ".pdf":
        return parse_pdf(file)

    if extension == ".docx":
        return parse_docx(file)

    if extension == ".txt":
        return parse_txt(file)

    raise DocumentParserError(
        f"Unsupported document type: {extension}"
    )


def parse_pdf(file: BinaryIO) -> List[Dict]:
    """
    Extract text page-by-page from a PDF.
    """

    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")

        pages = []

        for page_number, page in enumerate(pdf, start=1):

            pages.append(
                {
                    "page_number": page_number,
                    "text": page.get_text("text")
                }
            )

        pdf.close()

        return pages

    except Exception as e:
        raise DocumentParserError(
            f"Failed to parse PDF: {e}"
        )


def parse_docx(file: BinaryIO) -> List[Dict]:
    """
    Extract text from a DOCX document.
    DOCX has no page information, so page_number = 1.
    """

    try:

        document = DocxDocument(BytesIO(file.read()))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

        return [
            {
                "page_number": 1,
                "text": text,
            }
        ]

    except Exception as e:
        raise DocumentParserError(
            f"Failed to parse DOCX: {e}"
        )


def parse_txt(file: BinaryIO) -> List[Dict]:
    """
    Extract text from a TXT document.
    """

    try:

        text = file.read().decode("utf-8")

        return [
            {
                "page_number": 1,
                "text": text,
            }
        ]

    except UnicodeDecodeError:

        try:

            file.seek(0)

            text = file.read().decode("latin-1")

            return [
                {
                    "page_number": 1,
                    "text": text,
                }
            ]

        except Exception as e:

            raise DocumentParserError(
                f"Failed to parse TXT: {e}"
            )

    except Exception as e:

        raise DocumentParserError(
            f"Failed to parse TXT: {e}"
        )