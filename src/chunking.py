"""
chunking.py

Main chunking pipeline.

Responsibilities:
- Iterate through document pages
- Split page text into chunks
- Generate chunk metadata
- Return standardized Chunk objects
"""

from typing import List

from src.chunk_models import Chunk
from src.models import Document
from src.splitter import RecursiveTextSplitter


class DocumentChunker:
    """
    Converts a Document into a list of semantic chunks.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
    ):

        self.splitter = RecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk_document(
        self,
        document: Document,
    ) -> List[Chunk]:
        """
        Split an entire document into chunks.

        Parameters
        ----------
        document : Document

        Returns
        -------
        List[Chunk]
        """

        chunks: List[Chunk] = []

        chunk_counter = 1

        for page in document.pages:

            page_chunks = self.splitter.split_text(page.text)

            for index, piece in enumerate(page_chunks):

                chunk = Chunk(
                    chunk_id=f"{document.metadata.document_id}_chunk_{chunk_counter:04d}",
                    document_id=document.metadata.document_id,
                    filename=document.metadata.filename,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=piece["text"],
                    start_char=piece["start_char"],
                    end_char=piece["end_char"],
                )

                chunks.append(chunk)

                chunk_counter += 1

        return chunks