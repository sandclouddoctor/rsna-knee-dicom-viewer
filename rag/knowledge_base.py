"""
Chunk the mri-interpreter knee knowledge markdown into retrievable text records,
and embed them with a sentence-transformer for the text side of the RAG.

Usage:
    from rag.knowledge_base import load_knowledge_chunks, embed_knowledge
    chunks = load_knowledge_chunks()
    ids, embeddings = embed_knowledge(chunks)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_MD = Path(__file__).parent / "knowledge" / "knee_mri_knowledge.md"


@dataclass
class KnowledgeChunk:
    chunk_id: str
    title: str
    text: str


def load_knowledge_chunks(md_path: Path = KNOWLEDGE_MD) -> list[KnowledgeChunk]:
    """Split the knowledge markdown on '## ' headings into one chunk per section."""
    raw = md_path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", raw)
    chunks = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section.startswith("## "):
            continue  # skip the H1 preamble
        title_line, *body_lines = section.splitlines()
        title = title_line.removeprefix("## ").strip()
        body = "\n".join(body_lines).strip()
        chunk_id = f"knee_kb_{i:03d}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}"
        chunks.append(KnowledgeChunk(chunk_id=chunk_id, title=title, text=body))
    return chunks


_MODEL = None


def _get_text_encoder():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        # Small, fast, good-enough general encoder; swap for a biomedical
        # model (e.g. pritamdeka/S-PubMedBert-MS-MARCO) if retrieval quality
        # on radiology text needs improvement.
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def embed_knowledge(chunks: list[KnowledgeChunk]):
    """Returns (chunk_ids, embeddings ndarray [n, d])."""
    encoder = _get_text_encoder()
    texts = [f"{c.title}. {c.text}" for c in chunks]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [c.chunk_id for c in chunks], embeddings


def embed_query_text(query: str):
    encoder = _get_text_encoder()
    return encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
