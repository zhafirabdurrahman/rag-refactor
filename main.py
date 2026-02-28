from fastapi import FastAPI

from api.router import create_router
from services.document_store import build_document_store
from services.embedding_service import EmbeddingService
from services.rag_workflow import RagWorkflow


def create_app() -> FastAPI:
    """Compose all dependencies and return a configured FastAPI application."""

    # Build services (innermost - outermost dependency order)
    embedding_service = EmbeddingService()
    document_store = build_document_store(embedding_service)
    rag_workflow = RagWorkflow(document_store)

    # Create FastAPI app and attach the router
    app = FastAPI(title="Learning RAG Demo")
    app.include_router(create_router(document_store, rag_workflow))

    return app


app = create_app()
