from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self.store = store
        self.llm_fn = llm_fn
        pass

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "No relevant context found in the knowledge base."

        context_blocks = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("source") or metadata.get("doc_id") or result.get("id", "unknown")
            context_blocks.append(f"[{index}] source={source}\n{result['content']}")

        prompt = (
            "Answer the question using only the context below. "
            "If the context is insufficient, say that the answer was not found in the provided context. "
            "Cite relevant chunk numbers like [1].\n\n"
            f"Context:\n{chr(10).join(context_blocks)}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self.llm_fn(prompt)
        raise NotImplementedError("Implement KnowledgeBaseAgent.answer")
