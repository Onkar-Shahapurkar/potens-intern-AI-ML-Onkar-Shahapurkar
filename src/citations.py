"""
citations.py

Utilities for formatting citations returned by the RAG system.
"""

from __future__ import annotations

from typing import Dict, List


class CitationFormatter:
    """
    Formats citations for RAG responses.
    """

    @staticmethod
    def format_citations(
        retrieved_chunks: List[Dict],
    ) -> List[Dict]:
        """
        Convert retrieved chunks into structured citations.
        """

        citations = []

        for chunk in retrieved_chunks:

            snippet = chunk["text"].strip()

            if len(snippet) > 250:
                snippet = snippet[:250].rstrip() + "..."

            citations.append(
                {
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "snippet": snippet,
                }
            )

        return citations

    @staticmethod
    def format_markdown(
        citations: List[Dict],
    ) -> str:
        """
        Convert citations into a readable Markdown string.
        """

        if not citations:
            return "No citations available."

        lines = ["## Sources\n"]

        for index, citation in enumerate(citations, start=1):

            lines.extend(
                [
                    f"### {index}. {citation['filename']}",
                    f"- **Page:** {citation['page_number']}",
                    f"- **Chunk:** `{citation['chunk_id']}`",
                    "",
                    "> " + citation["snippet"],
                    "",
                ]
            )

        return "\n".join(lines)

    @staticmethod
    def format_json(
        citations: List[Dict],
    ) -> Dict:
        """
        Return citations in JSON format.
        """

        return {
            "total_sources": len(citations),
            "citations": citations,
        }