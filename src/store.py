from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401

            # TODO: initialize chromadb client + collection
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        metadata = dict(doc.metadata or {})
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }
        raise NotImplementedError("Implement EmbeddingStore._make_record")

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        if top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)
        results: list[dict[str, Any]] = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            results.append(
                {
                    "id": record["id"],
                    "content": record["content"],
                    "metadata": dict(record["metadata"]),
                    "score": score,
                }
            )
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]
        raise NotImplementedError("Implement EmbeddingStore._search_records")

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        records = [self._make_record(doc) for doc in docs]
        if not records:
            return

        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []
            for record in records:
                stored_id = f"{record['id']}::{self._next_index}"
                self._next_index += 1
                ids.append(stored_id)
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                metadatas.append(record["metadata"])
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            return

        self._store.extend(records)
        return
        raise NotImplementedError("Implement EmbeddingStore.add_documents")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            response = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
            ids = response.get("ids", [[]])[0]
            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0] if response.get("distances") else [0.0] * len(ids)
            return [
                {
                    "id": ids[index],
                    "content": documents[index],
                    "metadata": dict(metadatas[index] or {}),
                    "score": -distances[index],
                }
                for index in range(len(ids))
            ]

        return self._search_records(query, self._store, top_k)
        raise NotImplementedError("Implement EmbeddingStore.search")

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma and self._collection is not None:
            return int(self._collection.count())
        return len(self._store)
        raise NotImplementedError("Implement EmbeddingStore.get_collection_size")

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            response = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=metadata_filter,
            )
            ids = response.get("ids", [[]])[0]
            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0] if response.get("distances") else [0.0] * len(ids)
            return [
                {
                    "id": ids[index],
                    "content": documents[index],
                    "metadata": dict(metadatas[index] or {}),
                    "score": -distances[index],
                }
                for index in range(len(ids))
            ]

        filtered_records = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)
        raise NotImplementedError("Implement EmbeddingStore.search_with_filter")

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma and self._collection is not None:
            before = self.get_collection_size()
            self._collection.delete(where={"doc_id": doc_id})
            return self.get_collection_size() < before

        before = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
        raise NotImplementedError("Implement EmbeddingStore.delete_document")
