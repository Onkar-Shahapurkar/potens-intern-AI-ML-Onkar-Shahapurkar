"""
streamlit_app.py

Phase 8:
RAG Question Answering with Citations
"""

import streamlit as st

from src.chunking import DocumentChunker
from src.citations import CitationFormatter
from src.indexing import DocumentIndexer
from src.ingestion import (
    DocumentIngestor,
    DocumentIngestionError,
)
from src.llm import LLMService
from src.retrieval import Retriever

st.set_page_config(
    page_title="POTENS AI/ML Assignment",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Document Q&A with Citations")

st.markdown(
    """
Upload a document, build the knowledge base,
and ask questions grounded in the uploaded documents.
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
    llm = LLMService()

    try:

        # ==================================================
        # INGESTION
        # ==================================================

        document = ingestor.ingest(
            file=uploaded_file,
            filename=uploaded_file.name,
        )

        st.success("✅ Document ingested successfully.")

        # ==================================================
        # CHUNKING
        # ==================================================

        chunks = chunker.chunk_document(document)

        # ==================================================
        # INDEXING
        # ==================================================

        with st.spinner(
            "Generating embeddings and indexing..."
        ):

            stats = indexer.index_chunks(chunks)

        st.success("Knowledge base created.")

        st.divider()

        st.subheader("Knowledge Base")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Chunks",
            stats["indexed_chunks"],
        )

        c2.metric(
            "Vectors",
            stats["total_vectors"],
        )

        c3.metric(
            "Pages",
            document.metadata.page_count,
        )

        st.write(
            f"**Embedding Model:** {stats['embedding_model']}"
        )

        st.write(
            f"**Collection:** `{stats['collection_name']}`"
        )

        # ==================================================
        # QUESTION ANSWERING
        # ==================================================

        st.divider()

        st.header("💬 Ask Questions")

        question = st.text_input(
            "Question",
            placeholder="Ask something about the uploaded document...",
        )

        top_k = st.slider(
            "Retrieved Chunks",
            1,
            10,
            5,
        )

        if st.button(
            "Generate Answer",
            type="primary",
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Searching documents..."
                ):

                    retrieval = retriever.retrieve_for_generation(
                        query=question,
                        top_k=top_k,
                    )

                context = retrieval["context"]
                retrieved_chunks = retrieval["chunks"]

                answer = llm.generate_answer(
                    question=question,
                    context=context,
                )

                citations = CitationFormatter.format_citations(
                    retrieved_chunks
                )

                # ==========================================
                # ANSWER
                # ==========================================

                st.divider()

                st.subheader("Answer")

                st.write(answer)

                # ==========================================
                # CITATIONS
                # ==========================================

                st.subheader("Sources")

                if citations:

                    for citation in citations:

                        with st.expander(
                            f"{citation['filename']} | Page {citation['page_number']}"
                        ):

                            st.write(
                                f"**Chunk ID:** `{citation['chunk_id']}`"
                            )

                            st.info(
                                citation["snippet"]
                            )

                else:

                    st.warning(
                        "No supporting citations available."
                    )

                # ==========================================
                # RETRIEVED CONTEXT
                # ==========================================

                with st.expander(
                    "Retrieved Context"
                ):

                    st.text(context)

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.info(
    """
### Current Progress

✅ Phase 4 — Document Ingestion

✅ Phase 5 — Chunking

✅ Phase 6 — Gemini Embeddings & ChromaDB

✅ Phase 7 — Semantic Retrieval

✅ Phase 8 — Question Answering with Citations

Next:
- 🔜 /contradict endpoint
- 🔜 Multilingual support
- 🔜 Confidence scoring
"""
)