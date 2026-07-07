"""
app.py

FastAPI application entry point.
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="POTENS AI/ML RAG API",
    description="Document Question Answering with Citations, Contradiction Analysis, and Multilingual Support.",
    version="1.0.0",
)

app.include_router(router)


@app.get("/")
def root():
    """
    Root endpoint.
    """
    return {
        "message": "POTENS AI/ML RAG API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    """
    Health check endpoint.
    """
    return {
        "status": "ok"
    }