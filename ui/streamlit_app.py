"""
streamlit_app.py

Phase 4:
Document Ingestion Interface
"""

from pathlib import Path

import streamlit as st

from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)

st.set_page_config(
    page_title="POTENS AI/ML Assignment",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Document Ingestion Pipeline")
st.markdown(
    """
Upload a document to test the ingestion pipeline.

**Supported Formats**
- PDF
- DOCX
- TXT
"""
)

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "docx", "txt"],
)

if uploaded_file is not None:

    ingestor = DocumentIngestor()

    try:

        document = ingestor.ingest(
            file=uploaded_file,
            filename=uploaded_file.name,
        )

        st.success("Document ingested successfully!")

        metadata = document.metadata

        st.subheader("Document Metadata")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pages", metadata.page_count)
            st.metric("Words", metadata.word_count)
            st.metric("Characters", metadata.character_count)

        with col2:
            st.write(f"**Filename:** {metadata.filename}")
            st.write(f"**Type:** {metadata.file_type}")
            st.write(f"**Language:** {metadata.language}")

        st.divider()

        st.subheader("Extracted Pages")

        for page in document.pages:

            with st.expander(f"Page {page.page_number}"):

                st.text(page.text)

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.caption(
    "Phase 4 • Document Ingestion • POTENS Internship 2026"
)