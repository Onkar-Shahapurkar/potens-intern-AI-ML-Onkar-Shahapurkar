"""
streamlit_app.py

Phase 7:
Document Ingestion + Chunking + Indexing + Semantic Retrieval
"""

import streamlit as st

from src.chunking import DocumentChunker
from src.indexing import DocumentIndexer
from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)
from src.retrieval import Retriever

st.set_page_config(
    page_title="POTENS AI/ML Assignment",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 RAG Document Retrieval System")

st.markdown(
    """
Upload a document, index it into the vector database,
and search it using semantic retrieval.
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
    retriever = Retriever()

    try:

        # ==================================================
        # STEP 1 : INGESTION
        # ==================================================

        document = ingestor.ingest(
            file=uploaded_file,
            filename=uploaded_file.name,
        )

        st.success("✅ Document ingested successfully.")

        metadata = document.metadata

        st.subheader("📄 Document Information")

        c1, c2, c3 = st.columns(3)

        c1.metric("Pages", metadata.page_count)
        c2.metric("Words", metadata.word_count)
        c3.metric("Characters", metadata.character_count)

        st.write(f"**Filename:** {metadata.filename}")
        st.write(f"**Document ID:** `{metadata.document_id}`")
        st.write(f"**Language:** {metadata.language}")

        # ==================================================
        # STEP 2 : CHUNKING
        # ==================================================

        chunks = chunker.chunk_document(document)

        st.divider()

        st.subheader("🧩 Chunking")

        st.metric("Generated Chunks", len(chunks))

        with st.expander("Preview Chunks"):

            for chunk in chunks:

                st.markdown(f"### Chunk {chunk.chunk_index}")

                st.write(f"**Page:** {chunk.page_number}")

                st.code(chunk.text)

        # ==================================================
        # STEP 3 : INDEXING
        # ==================================================

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
                "Total Vectors",
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

        st.success("Knowledge base created successfully!")

        # ==================================================
        # STEP 4 : RETRIEVAL
        # ==================================================

        st.divider()

        st.subheader("🔍 Semantic Search")

        query = st.text_input(
            "Ask a retrieval query",
            placeholder="Example: What is Retrieval-Augmented Generation?",
        )

        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=10,
            value=5,
        )

        if st.button("Search"):

            with st.spinner("Searching..."):

                results = retriever.retrieve(
                    query=query,
                    top_k=top_k,
                )

            if not results:

                st.warning("No relevant chunks found.")

            else:

                st.success(
                    f"Retrieved {len(results)} chunk(s)."
                )

                for i, result in enumerate(results, start=1):

                    with st.expander(f"Result {i}"):

                        st.write(
                            f"**Chunk ID:** `{result['chunk_id']}`"
                        )

                        st.write(
                            f"**Filename:** {result['filename']}"
                        )

                        st.write(
                            f"**Page:** {result['page_number']}"
                        )

                        st.write(
                            f"**Distance:** {result['distance']:.4f}"
                        )

                        st.write(
                            f"**Character Range:** "
                            f"{result['start_char']} → "
                            f"{result['end_char']}"
                        )

                        st.code(result["text"])

                st.divider()

                st.subheader("📄 Combined Retrieval Context")

                context = retriever.retrieve_context(
                    query=query,
                    top_k=top_k,
                )

                st.text_area(
                    "Context",
                    value=context,
                    height=300,
                )

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.info(
    """
### Current Progress

✅ Phase 4 — Document Ingestion

✅ Phase 5 — Document Chunking

✅ Phase 6 — Gemini Embeddings & ChromaDB

✅ Phase 7 — Semantic Retrieval

🔜 Next Phase:
LLM-powered Question Answering with Citations (`/ask`)
"""
)