"""
ui/pages/upload.py

Upload and index documents into the knowledge base.
"""

from __future__ import annotations

import streamlit as st

from src.chunking import DocumentChunker
from src.indexing import DocumentIndexer
from src.ingestion import (
    DocumentIngestionError,
    DocumentIngestor,
)
from ui.components.chunk_preview import render_chunk_preview
from ui.components.document_table import render_document_table


from ui.services import (
    get_chunker,
    get_indexer,
    get_ingestor,
)


def initialize_session_state() -> None:
    """
    Initialize shared session state.
    """

    defaults = {
        "knowledge_base_ready": False,
        "indexed_documents": [],
        "uploaded_document_ids": set(),
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_upload_page() -> None:
    """
    Upload documents and build the knowledge base.
    """

    initialize_session_state()

    st.header("📄 Upload Documents")

    st.write(
        "Upload one or more documents to build the knowledge base."
    )

    uploaded_files = st.file_uploader(
        "Supported formats: PDF, DOCX, TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:

        if st.session_state["indexed_documents"]:
            render_document_table(
                st.session_state["indexed_documents"]
            )

        return

    ingestor = get_ingestor()
    chunker = get_chunker()
    indexer = get_indexer()

    if st.button(
        "🚀 Build Knowledge Base",
        type="primary",
        use_container_width=True,
    ):

        progress = st.progress(0.0)

        latest_chunks = []

        try:

            for index, uploaded_file in enumerate(uploaded_files):

                document = ingestor.ingest(
                    file=uploaded_file,
                    filename=uploaded_file.name,
                )

                document_id = document.metadata.document_id

                if (
                    document_id
                    in st.session_state[
                        "uploaded_document_ids"
                    ]
                ):
                    continue

                chunks = chunker.chunk_document(
                    document
                )

                latest_chunks = chunks

                indexer.index_chunks(chunks)

                st.session_state[
                    "uploaded_document_ids"
                ].add(document_id)

                st.session_state[
                    "indexed_documents"
                ].append(
                    {
                        "id": document_id,
                        "filename": document.metadata.filename,
                        "pages": document.metadata.page_count,
                        "chunks": len(chunks),
                    }
                )

                progress.progress(
                    (index + 1)
                    / len(uploaded_files)
                )

            st.session_state[
                "knowledge_base_ready"
            ] = True

            st.success(
                "Knowledge base built successfully."
            )

            render_document_table(
                st.session_state[
                    "indexed_documents"
                ]
            )

            if latest_chunks:

                render_chunk_preview(
                    latest_chunks
                )

        except DocumentIngestionError as exc:

            st.error(str(exc))

        except Exception as exc:

            st.exception(exc)

    if st.session_state["indexed_documents"]:

        st.divider()

        render_document_table(
            st.session_state["indexed_documents"]
        )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.session_state[
            "knowledge_base_ready"
        ]:

            st.success(
                "✅ Knowledge base is ready."
            )

        else:

            st.info(
                "Upload documents to begin."
            )

    with col2:

        if st.button(
            "🗑️ Reset Knowledge Base",
            use_container_width=True,
        ):

            st.session_state["knowledge_base_ready"] = False
            st.session_state["indexed_documents"] = []
            st.session_state["uploaded_document_ids"] = set()

            get_indexer().reset_index()

            st.rerun()