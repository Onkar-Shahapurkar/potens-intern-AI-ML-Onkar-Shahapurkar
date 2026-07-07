"""
ui/pages/contradiction.py

Document contradiction analysis page.
"""

from __future__ import annotations

import streamlit as st

from ui.services import get_contradiction


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

    indexed_documents = st.session_state.get(
        "indexed_documents",
        [],
    )

    if len(indexed_documents) < 2:

        st.info(
            "Please upload at least two documents to perform contradiction analysis."
        )
        return

    analyzer = get_contradiction()

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

        with st.spinner(
            "Analyzing documents..."
        ):

            result = analyzer.analyze(
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

        st.subheader("Analysis Result")

        if result["conflict"]:

            st.error(
                "⚠️ Contradiction Detected"
            )

        else:

            st.success(
                "✅ No Contradiction Detected"
            )

        st.markdown("### Reason")

        st.write(result["reason"])

        if result.get("evidence"):

            st.markdown("### Supporting Evidence")

            for idx, evidence in enumerate(
                result["evidence"],
                start=1,
            ):

                with st.expander(
                    f"Evidence {idx}"
                ):

                    st.write(
                        f"**Document:** {evidence.get('document', '-')}"
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
                result.get(
                    "document_a",
                    document_map[document_a],
                ),
            )

        with col2:

            st.metric(
                "Document B",
                result.get(
                    "document_b",
                    document_map[document_b],
                ),
            )