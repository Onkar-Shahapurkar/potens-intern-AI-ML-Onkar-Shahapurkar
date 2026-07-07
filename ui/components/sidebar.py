"""
ui/components/sidebar.py

Sidebar component for the Streamlit application.
"""

from __future__ import annotations

import streamlit as st


def render_sidebar() -> None:
    """
    Render the application sidebar.
    """

    with st.sidebar:

        st.title("🤖 POTENS RAG")

        st.caption(
            "AI-Powered Document Question Answering"
        )

        st.divider()

        ready = st.session_state.get(
            "knowledge_base_ready",
            False,
        )

        if ready:

            st.success("Knowledge Base Ready")

            documents = st.session_state.get(
                "indexed_documents",
                []
            )

            st.metric(
                "Documents",
                len(documents),
            )

            total_chunks = sum(
                doc.get("chunks", 0)
                for doc in documents
            )

            st.metric(
                "Indexed Chunks",
                total_chunks,
            )

        else:

            st.warning(
                "No documents indexed."
            )

        st.divider()

        st.markdown("### Features")

        st.markdown(
            """
- 📄 Document Upload
- 🧩 Chunking
- 🔍 Semantic Retrieval
- 💬 Question Answering
- 📑 Citations
- ⚖️ Contradiction Analysis
- 🌍 Multilingual Support
"""
        )

        st.divider()

        st.markdown("### Tech Stack")

        st.caption(
            """
• Gemini 2.5 Flash

• Gemini Embeddings

• ChromaDB

• FastAPI

• Streamlit
"""
        )

        st.divider()

        if st.button(
            "🗑️ Clear Session",
            use_container_width=True,
        ):

            for key in list(st.session_state.keys()):

                del st.session_state[key]

            st.rerun()