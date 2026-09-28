"""
Combined retrieval: given a new study, return (a) similar prior training cases
by DINOv3 image embedding, and (b) relevant mri-interpreter knowledge chunks by
text embedding of a query (a finding name, a free-text question, or the
retrieval-augmented classifier's own draft findings — see classifier.py).

This is the single entry point both usage modes from your brief go through:
  - "knowledge/reference tool": call retrieve_knowledge(query) alone
  - "retrieval-augmented classifier": call retrieve_similar_cases(embedding)
    then feed the results into classifier.py
"""
from __future__ import annotations

from pathlib import Path

from .knowledge_base import embed_knowledge, embed_query_text, load_knowledge_chunks
from .vector_store import VectorStore

CASE_INDEX_PATH = "model_ensemble/index/cases"
KNOWLEDGE_INDEX_PATH = "model_ensemble/index/knowledge"


def build_knowledge_index(save_path: str = KNOWLEDGE_INDEX_PATH) -> VectorStore:
    chunks = load_knowledge_chunks()
    ids, embeddings = embed_knowledge(chunks)
    store = VectorStore(dim=embeddings.shape[1])
    metadata = [{"title": c.title, "text": c.text} for c in chunks]
    store.add(ids, embeddings, metadata)
    store.save(save_path)
    return store


def build_case_index(study_embeddings: dict, study_metadata: dict, save_path: str = CASE_INDEX_PATH) -> VectorStore:
    """study_embeddings: {study_uid: np.ndarray}. study_metadata: {study_uid: {labels: {...}, report: str}}."""
    import numpy as np

    ids = list(study_embeddings.keys())
    dim = next(iter(study_embeddings.values())).shape[0]
    store = VectorStore(dim=dim)
    embeddings = np.stack([study_embeddings[i] for i in ids])
    metadata = [study_metadata.get(i, {}) for i in ids]
    store.add(ids, embeddings, metadata)
    store.save(save_path)
    return store


def retrieve_knowledge(query: str, k: int = 3, index_path: str = KNOWLEDGE_INDEX_PATH):
    if not Path(index_path).exists():
        build_knowledge_index(index_path)
    store = VectorStore.load(index_path)
    query_emb = embed_query_text(query)
    return store.search(query_emb, k=k)


def retrieve_similar_cases(study_embedding, k: int = 5, index_path: str = CASE_INDEX_PATH):
    if not Path(index_path).exists():
        raise FileNotFoundError(
            f"No case index at {index_path} — run build_case_index() on the pilot "
            f"batch (or a larger set once you're ready) before calling this."
        )
    store = VectorStore.load(index_path)
    return store.search(study_embedding, k=k)
