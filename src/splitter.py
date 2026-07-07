"""
splitter.py

Wrapper around LangChain's RecursiveCharacterTextSplitter.

Responsibilities:
- Split text into overlapping chunks
- Preserve character offsets
- Return a standardized structure for the chunking pipeline
"""

from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveTextSplitter:
    """
    Production wrapper around LangChain's
    RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                "? ",
                "! ",
                " ",
                "",
            ],
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(
        self,
        text: str,
    ) -> List[Dict]:
        """
        Split text into chunks while preserving
        approximate character offsets.
        """

        if not text.strip():
            return []

        texts = self.splitter.split_text(text)

        chunks = []

        current_position = 0

        for chunk in texts:

            start = text.find(chunk, current_position)

            if start == -1:
                start = current_position

            end = start + len(chunk)

            chunks.append(
                {
                    "text": chunk,
                    "start_char": start,
                    "end_char": end,
                }
            )

            current_position = end

        return chunks