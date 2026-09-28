"""
Thin FAISS wrappers for the two indices the RAG needs:
  - image index: DINOv3 study embeddings -> similar prior training cases
  - text index: sentence-transformer embeddings of mri-interpreter knowledge chunks

Kept deliberately simple (flat, exact search) since both indices are small
(pilot batch of ~30 studies, ~15 knowledge chunks) — swap for IndexIVFFlat/HNSW
only if you scale the case index up to the full 4407 studies.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # cosine similarity, since we L2-normalize
        self.ids: list[str] = []
        self.metadata: dict[str, dict] = {}

    def add(self, ids: list[str], embeddings: np.ndarray, metadata: list[dict] | None = None):
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.ids.extend(ids)
        if metadata:
            for _id, meta in zip(ids, metadata):
                self.metadata[_id] = meta

    def search(self, query_embedding: np.ndarray, k: int = 5):
        query = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query, min(k, len(self.ids)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            _id = self.ids[idx]
            results.append({"id": _id, "score": float(score), "metadata": self.metadata.get(_id, {})})
        return results

    def save(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "meta.pkl", "wb") as f:
            pickle.dump({"ids": self.ids, "metadata": self.metadata, "dim": self.dim}, f)

    @classmethod
    def load(cls, path: str) -> "VectorStore":
        with open(Path(path) / "meta.pkl", "rb") as f:
            state = pickle.load(f)
        store = cls(dim=state["dim"])
        store.index = faiss.read_index(str(Path(path) / "index.faiss"))
        store.ids = state["ids"]
        store.metadata = state["metadata"]
        return store
