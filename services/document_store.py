import uuid

from abc import ABC, abstractmethod
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from services.embedding_service import EmbeddingService


class DocumentStore(ABC):
    """Abstract interface for document persistence and retrieval."""

    @abstractmethod
    def add(self, text: str) -> str:
        """Persist text and return its assigned ID."""

    @abstractmethod
    def search(self, query: str, limit: int = 2) -> list[str]:
        """Return up to limit documents most relevant to query."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable name of the storage backend."""


# Qdrant implementation
class QdrantDocumentStore(DocumentStore):
    _COLLECTION = "demo_collection"

    def __init__(self, client: QdrantClient, embedding_service: EmbeddingService):
        self._client = client
        self._embedder = embedding_service
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        self._client.recreate_collection(
            collection_name=self._COLLECTION,
            vectors_config=VectorParams(
                size=EmbeddingService.VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

    def add(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        vector = self._embedder.embed(text)
        self._client.upsert(
            collection_name=self._COLLECTION,
            points=[PointStruct(id=doc_id, vector=vector, payload={"text": text})],
        )
        return doc_id

    def search(self, query: str, limit: int = 2) -> list[str]:
        vector = self._embedder.embed(query)
        hits = self._client.search(
            collection_name=self._COLLECTION,
            query_vector=vector,
            limit=limit,
        )
        return [hit.payload["text"] for hit in hits]

    @property
    def backend_name(self) -> str:
        return "qdrant"


# In-memory fallback
class InMemoryDocumentStore(DocumentStore):
    def __init__(self) -> None:
        self._docs: list[str] = []

    def add(self, text: str) -> str:
        doc_id = str(len(self._docs))
        self._docs.append(text)
        return doc_id

    def search(self, query: str, limit: int = 2) -> list[str]:
        matches = [d for d in self._docs if query.lower() in d.lower()]
        if not matches and self._docs:
            matches = [self._docs[0]]
        return matches[:limit]

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    @property
    def backend_name(self) -> str:
        return "in_memory"


# Factory
def build_document_store(
    embedding_service: EmbeddingService,
    qdrant_url: str = "http://localhost:6333",
) -> DocumentStore:
    """Try to connect to Qdrant, fall back to in-memory if unavailable."""
    try:
        client = QdrantClient(qdrant_url)
        store = QdrantDocumentStore(client, embedding_service)
        print("Connected to Qdrant.")
        return store
    except Exception:
        print("Qdrant not available, using in-memory store.")
        return InMemoryDocumentStore()
