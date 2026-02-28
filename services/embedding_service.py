import random


class EmbeddingService:
    """Generates vector embeddings for text.

    Currently uses a deterministic fake implementation.
    Replace `embed()` with a real model call (e.g. OpenAI, SentenceTransformers)
    without touching any other layer.
    """

    VECTOR_SIZE = 128

    def embed(self, text: str) -> list[float]:
        """Return a deterministic pseudo-random embedding vector for *text*."""
        random.seed(abs(hash(text)) % 10_000)
        return [random.random() for _ in range(self.VECTOR_SIZE)]
