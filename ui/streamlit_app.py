"""
streamlit_app.py

Phase 9:
RAG Document Q&A + Contradiction Analysis
"""

import streamlit as st

from src.chunking import DocumentChunker
from src.citations import CitationFormatter
from src.contradiction import ContradictionAnalyzer
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

uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if uploaded_files:

    ingestor = DocumentIngestor()
    chunker = DocumentChunker()
    indexer = DocumentIndexer()
    retriever = Retriever()
    llm = LLMService()
    contradiction = ContradictionAnalyzer(
        retriever=retriever,
        llm=llm,
    )

    indexed_documents = []

    try:

        progress = st.progress(0)

        for i, uploaded_file in enumerate(uploaded_files):

            document = ingestor.ingest(
                file=uploaded_file,
                filename=uploaded_file.name,
            )

            chunks = chunker.chunk_document(document)

            indexer.index_chunks(chunks)

            indexed_documents.append(
                {
                    "id": document.metadata.document_id,
                    "filename": document.metadata.filename,
                    "pages": document.metadata.page_count,
                    "chunks": len(chunks),
                }
            )

            progress.progress((i + 1) / len(uploaded_files))

        st.success(
            f"Indexed {len(indexed_documents)} document(s)."
        )

        st.divider()

        st.subheader("Indexed Documents")

        st.dataframe(
            indexed_documents,
            use_container_width=True,
        )

        # =====================================================
        # QUESTION ANSWERING
        # =====================================================

        st.divider()

        st.header("💬 Ask Questions")

        question = st.text_input(
            "Question",
            placeholder="Ask something about the uploaded documents...",
        )

        top_k = st.slider(
            "Top K Retrieval",
            1,
            10,
            5,
        )

        if st.button(
            "Generate Answer",
            type="primary",
        ):

            retrieval = retriever.retrieve_for_generation(
                query=question,
                top_k=top_k,
            )

            answer = llm.generate_answer(
                question=question,
                context=retrieval["context"],
            )

            citations = CitationFormatter.format_citations(
                retrieval["chunks"]
            )

            st.subheader("Answer")

            st.write(answer)

            st.subheader("Sources")

            if citations:

                for citation in citations:

                    with st.expander(
                        f"{citation['filename']} | Page {citation['page_number']}"
                    ):

                        st.write(
                            f"Chunk: `{citation['chunk_id']}`"
                        )

                        st.info(
                            citation["snippet"]
                        )

            else:

                st.warning(
                    "No citations available."
                )

        # =====================================================
        # CONTRADICTION ANALYSIS
        # =====================================================

        st.divider()

        st.header("⚖️ Document Contradiction Analysis")

        document_map = {
            f"{doc['filename']} ({doc['id']})": doc["id"]
            for doc in indexed_documents
        }

        col1, col2 = st.columns(2)

        with col1:

            document_a = st.selectbox(
                "Document A",
                options=list(document_map.keys()),
            )

        with col2:

            document_b = st.selectbox(
                "Document B",
                options=list(document_map.keys()),
                index=min(
                    1,
                    len(document_map) - 1,
                ),
            )

        topic = st.text_input(
            "Comparison Topic",
            placeholder="Example: Password Policy",
        )

        if st.button(
            "Analyze Contradictions"
        ):

            result = contradiction.analyze(
                topic=topic,
                document_a_id=document_map[
                    document_a
                ],
                document_b_id=document_map[
                    document_b
                ],
            )

            st.subheader("Analysis")

            if result["conflict"]:

                st.error(
                    "⚠️ Contradiction Detected"
                )

            else:

                st.success(
                    "✅ No Contradiction Detected"
                )

            st.write(
                f"**Reason:** {result['reason']}"
            )

            if result["evidence"]:

                st.subheader("Evidence")

                for item in result["evidence"]:

                    with st.expander(
                        item.get(
                            "document",
                            "Evidence",
                        )
                    ):

                        st.write(
                            item.get(
                                "snippet",
                                "",
                            )
                        )

    except DocumentIngestionError as e:

        st.error(str(e))

    except Exception as e:

        st.exception(e)

st.divider()

st.info(
    """
### Features Implemented

✅ Document Ingestion

✅ Document Chunking

✅ Gemini Embeddings

✅ ChromaDB Indexing

✅ Semantic Retrieval

✅ Question Answering with Citations

✅ Document Contradiction Analysis

Next:
- 🌍 Multilingual Query Support
- 📊 Confidence Score
- 🔁 Reranker
- 📈 Evaluation Pipeline
"""
)