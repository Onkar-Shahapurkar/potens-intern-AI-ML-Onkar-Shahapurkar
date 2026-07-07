"""
ui/pages/qa.py

Question Answering page.

Features
--------
- Multilingual query support
- Semantic retrieval
- RAG answer generation
- Source citations
"""

from __future__ import annotations

import streamlit as st

from src.citations import CitationFormatter
from ui.services import (
    get_llm,
    get_retriever,
)


def render_qa_page() -> None:
    """
    Render the Question Answering page.
    """

    st.header("💬 Question Answering")

    if not st.session_state.get(
        "knowledge_base_ready",
        False,
    ):
        st.warning(
            "Please upload and index documents first."
        )
        return

    retriever = get_retriever()
    llm = get_llm()

    question = st.text_area(
        "Ask a question",
        placeholder=(
            "Example:\n"
            "• What is Retrieval-Augmented Generation?\n"
            "• रिट्रिव्हल ऑगमेंटेड जनरेशन म्हणजे काय?\n"
            "• रिट्रीवल ऑगमेंटेड जनरेशन क्या है?"
        ),
        height=120,
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button(
        "🚀 Generate Answer",
        type="primary",
        use_container_width=True,
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return

        with st.spinner(
            "Searching documents..."
        ):

            retrieval = retriever.retrieve_for_generation(
                query=question,
                top_k=top_k,
            )

        context = retrieval["context"]

        retrieved_chunks = retrieval["chunks"]

        detected_language = retrieval["language"]

        translated_query = retrieval[
            "translated_query"
        ]

        with st.spinner(
            "Generating answer..."
        ):

            answer = llm.generate_answer(
                question=question,
                context=context,
            )

        citations = CitationFormatter.format_citations(
            retrieved_chunks
        )

        # ==========================================
        # Language
        # ==========================================

        st.divider()

        st.subheader("🌍 Language")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Detected",
                detected_language.upper(),
            )

        with col2:

            st.metric(
                "Retrieved Chunks",
                len(retrieved_chunks),
            )

        if detected_language != "en":

            st.caption(
                "Translated query used for retrieval:"
            )

            st.code(translated_query)

        # ==========================================
        # Answer
        # ==========================================

        st.divider()

        st.subheader("📝 Answer")

        st.write(answer)

        # ==========================================
        # Citations
        # ==========================================

        st.divider()

        st.subheader("📚 Sources")

        if not citations:

            st.warning(
                "No citations available."
            )

        else:

            for citation in citations:

                with st.expander(
                    f"{citation['filename']} • Page {citation['page_number']}"
                ):

                    st.write(
                        f"**Chunk ID:** `{citation['chunk_id']}`"
                    )

                    st.info(
                        citation["snippet"]
                    )

        # ==========================================
        # Retrieved Context
        # ==========================================

        with st.expander(
            "🔍 Retrieved Context"
        ):

            st.text(context)