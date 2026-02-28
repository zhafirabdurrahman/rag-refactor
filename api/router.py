import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from services.document_store import DocumentStore, InMemoryDocumentStore
from services.rag_workflow import RagWorkflow

# Request
class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    context_used: list[str]
    latency_sec: float

class DocumentRequest(BaseModel):
    text: str

class DocumentResponse(BaseModel):
    id: str
    status: str

class StatusResponse(BaseModel):
    backend: str
    in_memory_docs_count: int
    graph_ready: bool

# Router factory
def create_router(
    document_store: DocumentStore,
    rag_workflow: RagWorkflow,
) -> APIRouter:
    """Return an APIRouter pre-wired with the given service dependencies."""

    router = APIRouter()

    @router.post("/ask", response_model=QuestionResponse)
    def ask_question(req: QuestionRequest):
        start = time.time()
        try:
            result = rag_workflow.run(req.question)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return QuestionResponse(
            question=req.question,
            answer=result["answer"],
            context_used=result.get("context", []),
            latency_sec=round(time.time() - start, 3),
        )

    @router.post("/add", response_model=DocumentResponse)
    def add_document(req: DocumentRequest):
        try:
            doc_id = document_store.add(req.text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return DocumentResponse(id=doc_id, status="added")

    @router.get("/status", response_model=StatusResponse)
    def status():
        in_memory_count = (
            document_store.doc_count
            if isinstance(document_store, InMemoryDocumentStore)
            else 0
        )
        return StatusResponse(
            backend=document_store.backend_name,
            in_memory_docs_count=in_memory_count,
            graph_ready=True,
        )

    return router
