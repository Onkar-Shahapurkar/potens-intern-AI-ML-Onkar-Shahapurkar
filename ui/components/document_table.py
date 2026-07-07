"""
ui/components/document_table.py

Displays indexed documents in a clean table.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_document_table(
    indexed_documents: list[dict],
) -> None:
    """
    Render the indexed document table.
    """

    st.subheader("📚 Indexed Documents")

    if not indexed_documents:

        st.info(
            "No documents have been indexed yet."
        )

        return

    dataframe = pd.DataFrame(indexed_documents)

    dataframe = dataframe.rename(
        columns={
            "id": "Document ID",
            "filename": "Filename",
            "pages": "Pages",
            "chunks": "Chunks",
        }
    )

    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Documents",
            len(indexed_documents),
        )

    with col2:

        total_pages = sum(
            doc["pages"]
            for doc in indexed_documents
        )

        st.metric(
            "Pages",
            total_pages,
        )

    with col3:

        total_chunks = sum(
            doc["chunks"]
            for doc in indexed_documents
        )

        st.metric(
            "Chunks",
            total_chunks,
        )