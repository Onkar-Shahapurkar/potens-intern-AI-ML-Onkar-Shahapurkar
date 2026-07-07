"""
streamlit_app.py

Phase 6:
Document Ingestion + Chunking + Vector Indexing
"""

import streamlit as st

from src.chunking import DocumentChunker
from src.indexing import DocumentIndexer
from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)

st.set_page_config(
    page_title="POTENS AI/ML Assignment",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Document Indexing")

st.markdown(
    """
Upload one or more documents to build your knowledge base.

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

if uploaded_file:

    ingestor = DocumentIngestor()
    chunker = DocumentChunker()
    indexer = DocumentIndexer()

    try:

        # ---------------------------------------------------
        # STEP 1 : INGEST
        # ---------------------------------------------------

        document = ingestor.ingest(
            file=uploaded_file,
            filename=uploaded_file.name,
        )

        st.success("✅ Document ingested successfully.")

        metadata = document.metadata

        st.subheader("📄 Document Metadata")

        c1, c2, c3 = st.columns(3)

        c1.metric("Pages", metadata.page_count)
        c2.metric("Words", metadata.word_count)
        c3.metric("Characters", metadata.character_count)

        st.write(f"**Filename:** {metadata.filename}")
        st.write(f"**Document ID:** `{metadata.document_id}`")
        st.write(f"**Language:** {metadata.language}")

        # ---------------------------------------------------
        # STEP 2 : CHUNKING
        # ---------------------------------------------------

        chunks = chunker.chunk_document(document)

        st.divider()

        st.subheader("🧩 Chunk Generation")

        st.metric("Generated Chunks", len(chunks))

        with st.expander("Preview Generated Chunks"):

            for chunk in chunks:

                st.markdown(
                    f"### Chunk {chunk.chunk_index}"
                )

                st.write(
                    f"**Page:** {chunk.page_number}"
                )

                st.write(
                    f"**Characters:** "
                    f"{chunk.start_char} → {chunk.end_char}"
                )

                st.code(chunk.text)

        # ---------------------------------------------------
        # STEP 3 : VECTOR INDEXING
        # ---------------------------------------------------

        with st.spinner(
            "Generating Gemini embeddings and indexing..."
        ):

            stats = indexer.index_chunks(chunks)

        st.divider()

        st.subheader("📚 Vector Database")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Indexed Chunks",
                stats["indexed_chunks"],
            )

            st.metric(
                "Total Indexed Vectors",
                stats["total_vectors"],
            )

        with col2:

            st.write(
                f"**Embedding Model:** "
                f"{stats['embedding_model']}"
            )

            st.write(
                f"**Collection:** "
                f"`{stats['collection_name']}`"
            )

        st.success(
            "Knowledge base successfully created!"
        )

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.info(
    """
Current Phase: **Phase 6**

Completed:

- ✅ Document Ingestion
- ✅ Chunk Generation
- ✅ Gemini Embeddings
- ✅ ChromaDB Indexing

Next:

- 🔜 Question Answering (/ask)
- 🔜 Citations
- 🔜 Hallucination Guard
"""
)