"""
ui/pages/contradiction.py

Document contradiction analysis page using the FastAPI backend.
"""

from __future__ import annotations

import requests
import streamlit as st

from ui.api_client import get_api_client


def render_contradiction_page() -> None:
    """
    Render the contradiction analysis page.
    """

    st.header("⚖️ Document Contradiction Analysis")

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

    indexed_documents = st.session_state.get(
        "indexed_documents",
        [],
    )

    if len(indexed_documents) < 2:

        st.info(
            "Upload at least two indexed documents."
        )

        return

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

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=5,
    )

    if st.button(
        "🔍 Analyze Contradictions",
        type="primary",
        use_container_width=True,
    ):

        if not topic.strip():

            st.warning(
                "Please enter a comparison topic."
            )

            return

        try:

            with st.spinner(
                "Analyzing..."
            ):

                result = api.contradict(
                    topic=topic,
                    document_a_id=document_map[
                        document_a
                    ],
                    document_b_id=document_map[
                        document_b
                    ],
                    top_k=top_k,
                )

            st.divider()

            st.subheader(
                "📊 Analysis Result"
            )

            if result["conflict"]:

                st.error(
                    "⚠️ Contradiction Detected"
                )

            else:

                st.success(
                    "✅ No Contradiction Detected"
                )

            st.markdown("### Reason")

            st.write(
                result["reason"]
            )

            if result["evidence"]:

                st.divider()

                st.subheader(
                    "📚 Supporting Evidence"
                )

                for idx, evidence in enumerate(
                    result["evidence"],
                    start=1,
                ):

                    with st.expander(
                        f"Evidence {idx}"
                    ):

                        st.write(
                            f"**Document:** "
                            f"{evidence.get('document', '-')}"
                        )

                        st.info(
                            evidence.get(
                                "snippet",
                                "No snippet available.",
                            )
                        )

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Document A",
                    result["document_a"],
                )

            with col2:

                st.metric(
                    "Document B",
                    result["document_b"],
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