"""
ui/pages/qa.py

Question Answering page using the FastAPI backend.
Displays confidence metrics and reranking status.
"""

from __future__ import annotations

import requests
import streamlit as st

from ui.api_client import get_api_client


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

    api = get_api_client()

    if not api.is_available():

        st.error(
            "FastAPI backend is not running.\n\n"
            "Start it using:\n"
            "`uvicorn api.app:app --reload`"
        )
        return

    # --------------------------------------------------
    # Pipeline Status
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        st.success("✅ Semantic Retrieval")

    with col2:
        st.success("✅ Reranking Enabled")

    st.caption(
        "Pipeline: Embeddings → ChromaDB → Reranker → Gemini"
    )

    st.divider()

    question = st.text_area(
        "Ask a Question",
        placeholder=(
            "Examples:\n"
            "• What is Retrieval-Augmented Generation?\n"
            "• रिट्रिव्हल ऑगमेंटेड जनरेशन म्हणजे काय?\n"
            "• रिट्रीवल ऑगमेंटेड जनरेशन क्या है?"
        ),
        height=120,
    )

    top_k = st.slider(
        "Final Top-K Chunks",
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

        try:

            with st.spinner(
                "Searching documents and reranking..."
            ):

                response = api.ask(
                    question=question,
                    top_k=top_k,
                )

            # --------------------------------------------------
            # Answer
            # --------------------------------------------------

            st.divider()

            st.subheader("📝 Answer")

            st.write(response["answer"])

            # --------------------------------------------------
            # Confidence
            # --------------------------------------------------

            st.divider()

            st.subheader("📊 Confidence")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Language",
                    response["language"].upper(),
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{response['confidence']}%",
                )

            with col3:
                st.metric(
                    "Level",
                    response["confidence_level"],
                )

            level = response["confidence_level"]

            if level == "High":

                st.success(
                    "🟢 High confidence"
                )

            elif level == "Medium":

                st.warning(
                    "🟡 Medium confidence"
                )

            else:

                st.error(
                    "🔴 Low confidence"
                )

            if response["human_review"]:

                st.warning(
                    f"⚠️ {response['review_message']}"
                )

            else:

                st.info(
                    response["review_message"]
                )

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            metrics = response["metrics"]

            st.divider()

            st.subheader("📈 Confidence Metrics")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Retrieval",
                f"{metrics['retrieval']}%",
            )

            c2.metric(
                "Coverage",
                f"{metrics['coverage']}%",
            )

            c3.metric(
                "Citations",
                f"{metrics['citations']}%",
            )

            c4.metric(
                "Chunks",
                f"{metrics['chunks']}%",
            )

            # --------------------------------------------------
            # Citations
            # --------------------------------------------------

            st.divider()

            st.subheader("📚 Citations")

            citations = response["citations"]

            if not citations:

                st.warning(
                    "No citations available."
                )

            else:

                for citation in citations:

                    with st.expander(
                        f"{citation['filename']} | "
                        f"Page {citation['page_number']}"
                    ):

                        st.write(
                            f"**Chunk ID:** `{citation['chunk_id']}`"
                        )

                        st.info(
                            citation["snippet"]
                        )

        except requests.HTTPError as exc:

            try:

                detail = exc.response.json().get(
                    "detail",
                    str(exc),
                )

            except Exception:

                detail = str(exc)

            st.error(detail)

        except Exception as exc:

            st.exception(exc)