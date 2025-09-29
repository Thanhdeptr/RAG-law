from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.utils import embedding_functions


@dataclass
class QueryResults:
    documents: List[str]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class ChromaStore:
    def __init__(self, persist_dir: str, collection_name: str = "legal_chunks") -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        ids = [f"doc-{self.collection.count()}-{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas if metadatas is not None else [{} for _ in texts],
            embeddings=embeddings,
        )

    def query_by_embedding(self, query_embedding: List[float], top_k: int = 5) -> QueryResults:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return QueryResults(documents=docs, metadatas=metas, distances=dists)

    def get_texts_where(
        self,
        where: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Fetch documents and metadatas by metadata filter using Chroma's collection.get.
        Returns (documents, metadatas).
        """
        res = self.collection.get(where=where, limit=limit)
        docs: List[str] = res.get("documents", []) or []
        metas: List[Dict[str, Any]] = res.get("metadatas", []) or []
        return docs, metas

