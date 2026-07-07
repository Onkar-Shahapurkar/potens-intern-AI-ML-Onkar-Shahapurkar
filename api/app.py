"""
api/app.py

FastAPI application entry point.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router

app = FastAPI(
    title="POTENS AI/ML RAG API",
    description=(
        "Production-ready RAG API for document question answering, "
        "citations, contradiction analysis, and multilingual support."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(router)


# ---------------------------------------------------------
# Root
# ---------------------------------------------------------

@app.get(
    "/",
    tags=["System"],
)
def root():
    """
    Root endpoint.
    """

    return {
        "application": "POTENS AI/ML RAG API",
        "version": "1.0.0",
        "status": "running",
        "documentation": "/docs",
    }