"""
ui/components/chunk_preview.py

Chunk preview component.
"""

from __future__ import annotations

import streamlit as st

from src.chunk_models import Chunk


def render_chunk_preview(
    chunks: list[Chunk],
    max_chunks: int = 5,
) -> None:
    """
    Render a preview of generated chunks.
    """

    st.subheader("🧩 Chunk Preview")

    if not chunks:

        st.info(
            "No chunks available."
        )

        return

    st.caption(
        f"Showing first {min(max_chunks, len(chunks))} of {len(chunks)} chunk(s)."
    )

    for chunk in chunks[:max_chunks]:

        with st.expander(
            f"Chunk {chunk.chunk_index + 1}"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Document ID:** `{chunk.document_id}`"
                )

                st.write(
                    f"**Chunk ID:** `{chunk.chunk_id}`"
                )

                st.write(
                    f"**Filename:** {chunk.filename}"
                )

            with col2:

                st.write(
                    f"**Page:** {chunk.page_number}"
                )

                st.write(
                    f"**Characters:** "
                    f"{chunk.start_char} → {chunk.end_char}"
                )

            st.text_area(
                "Chunk Text",
                value=chunk.text,
                height=180,
                disabled=True,
                key=f"chunk_{chunk.chunk_id}",
            )

    if len(chunks) > max_chunks:

        st.info(
            f"{len(chunks) - max_chunks} additional chunk(s) not shown."
        )