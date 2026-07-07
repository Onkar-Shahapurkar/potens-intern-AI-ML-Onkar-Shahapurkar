"""
streamlit_app.py

Phase 5:
Document Ingestion + Chunking
"""

import streamlit as st

from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)
from src.chunking import DocumentChunker

st.set_page_config(
    page_title="POTENS AI/ML Assignment",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG Document Processing")

st.markdown(
    """
Upload a document to test the ingestion and chunking pipeline.

### Supported Formats
- PDF
- DOCX
- TXT
"""
)

uploaded_file = st.file_uploader(
    "Upload Document",
    type=["pdf", "docx", "txt"],
)

if uploaded_file is not None:

    ingestor = DocumentIngestor()
    chunker = DocumentChunker()

    try:

        # -----------------------------
        # Document Ingestion
        # -----------------------------
        document = ingestor.ingest(
            file=uploaded_file,
            filename=uploaded_file.name,
        )

        st.success("Document ingested successfully!")

        metadata = document.metadata

        st.header("📑 Document Metadata")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Pages", metadata.page_count)
            st.metric("Words", metadata.word_count)
            st.metric("Characters", metadata.character_count)

        with col2:
            st.write(f"**Filename:** {metadata.filename}")
            st.write(f"**Document ID:** {metadata.document_id}")
            st.write(f"**Language:** {metadata.language}")

        # -----------------------------
        # Page Preview
        # -----------------------------
        st.divider()

        st.header("📄 Extracted Pages")

        for page in document.pages:

            with st.expander(f"Page {page.page_number}"):

                st.text(page.text)

        # -----------------------------
        # Chunk Generation
        # -----------------------------
        st.divider()

        chunks = chunker.chunk_document(document)

        st.header("🧩 Generated Chunks")

        st.metric("Total Chunks", len(chunks))

        for chunk in chunks:

            with st.expander(f"Chunk {chunk.chunk_index}"):

                st.write(f"**Chunk ID:** `{chunk.chunk_id}`")
                st.write(f"**Page:** {chunk.page_number}")
                st.write(
                    f"**Characters:** "
                    f"{chunk.start_char} → {chunk.end_char}"
                )

                st.code(chunk.text)

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.caption(
    "Phase 5 • Document Ingestion + Chunking • POTENS Internship 2026"
)