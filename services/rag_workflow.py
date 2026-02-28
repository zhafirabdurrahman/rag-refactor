from typing import TypedDict
from langgraph.graph import END, StateGraph
from services.document_store import DocumentStore

class RagState(TypedDict):
    question: str
    context: list[str]
    answer: str

class RagWorkflow:
    """Encapsulates the LangGraph retrieve-answer pipeline.

    Accepts a DocumentStore so the retrieval strategy can be swapped or
    mocked without rebuilding the graph.
    """

    def __init__(self, document_store: DocumentStore) -> None:
        self._store = document_store
        self._chain = self._build_chain()

    # Graph nodes
    def _retrieve(self, state: RagState) -> RagState:
        results = self._store.search(state["question"])
        return {**state, "context": results}

    def _answer(self, state: RagState) -> RagState:
        ctx = state.get("context", [])
        answer = (
            f"I found this: '{ctx[0][:100]}...'" if ctx else "Sorry, I don't know."
        )
        return {**state, "answer": answer}

    # Graph construction
    def _build_chain(self):
        graph = StateGraph(RagState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("answer", self._answer)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    # Public interface
    def run(self, question: str) -> dict:
        """Execute the RAG pipeline and return the full result state."""
        return self._chain.invoke(
            {"question": question, "context": [], "answer": ""}
        )
