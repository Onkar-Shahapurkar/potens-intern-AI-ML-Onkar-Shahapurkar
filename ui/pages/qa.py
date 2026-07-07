"""
ui/pages/qa.py

Question Answering page using the FastAPI backend.
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

    question = st.text_area(
        "Ask a Question",
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

        try:

            with st.spinner(
                "Generating answer..."
            ):

                response = api.ask(
                    question=question,
                    top_k=top_k,
                )

            st.divider()

            st.subheader("📝 Answer")

            st.write(response["answer"])

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Language",
                    response["language"].upper(),
                )

            with col2:

                st.metric(
                    "Citations",
                    len(response["citations"]),
                )

            st.divider()

            st.subheader("📚 Citations")

            if not response["citations"]:

                st.info(
                    "No citations available."
                )

            else:

                for citation in response["citations"]:

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