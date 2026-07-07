"""
ui/streamlit_app.py

Main Streamlit application entry point.
"""

from __future__ import annotations

import streamlit as st

from ui.components.sidebar import render_sidebar
from ui.pages.contradiction import render_contradiction_page
from ui.pages.qa import render_qa_page
from ui.pages.upload import render_upload_page

st.set_page_config(
    page_title="POTENS AI/ML RAG System",
    page_icon="🤖",
    layout="wide",
)

render_sidebar()

st.title("🤖 POTENS AI/ML RAG System")

st.caption(
    "Document Question Answering • Citations • Contradiction Analysis • Multilingual Support"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "📄 Upload Documents",
        "💬 Question Answering",
        "⚖️ Contradiction Analysis",
    ],
)

if page == "📄 Upload Documents":
    render_upload_page()

elif page == "💬 Question Answering":
    render_qa_page()

elif page == "⚖️ Contradiction Analysis":
    render_contradiction_page()