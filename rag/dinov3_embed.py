"""
DINOv3 image embedding extraction for knee MRI slices.

Each training case is represented by a small set of embeddings (one per
representative slice, or a mean-pooled per-series embedding) so that the RAG's
image index can retrieve "cases that look like this one" at inference time.

Model: facebook/dinov3-* via transformers (falls back to dinov2 if dinov3
weights aren't available in your environment/transformers version — check
Kaggle's transformers version before relying on this on a GPU kernel).

This module intentionally does NOT run on the full 4407-study corpus by
default — call build_index_for_studies() with an explicit study list (start
with the pilot batch in data/pilot_studies.csv) to control compute/GPU-quota
spend.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_MODEL = None
_PROCESSOR = None
_DEVICE = None


def _get_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
        try:
            _PROCESSOR = AutoImageProcessor.from_pretrained(model_name)
            _MODEL = AutoModel.from_pretrained(model_name).to(_DEVICE).eval()
        except Exception:
            # Fallback for environments without DINOv3 weights available yet.
            fallback = "facebook/dinov2-small"
            _PROCESSOR = AutoImageProcessor.from_pretrained(fallback)
            _MODEL = AutoModel.from_pretrained(fallback).to(_DEVICE).eval()
    return _MODEL, _PROCESSOR, _DEVICE


def embed_image_array(arr: np.ndarray) -> np.ndarray:
    """arr: 2D float array in [0, 1] (grayscale MRI slice, already windowed).
    Returns a single L2-normalized embedding vector."""
    import torch

    model, processor, device = _get_model()
    rgb = np.stack([arr, arr, arr], axis=-1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token (or mean-pooled patch tokens as a robust fallback)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        else:
            emb = outputs.last_hidden_state[0].mean(dim=0)
    emb = emb.cpu().numpy()
    return emb / (np.linalg.norm(emb) + 1e-8)


def embed_study(series_list, max_slices_per_series: int = 3) -> np.ndarray:
    """Mean-pool embeddings across a handful of representative slices per
    series (avoids embedding every slice of every series, which would be
    wasteful for a case-similarity index)."""
    from viewer.dicom_utils import load_slice_array, window_image

    vectors = []
    for series in series_list:
        n = len(series.slice_paths)
        if n == 0:
            continue
        step = max(n // max_slices_per_series, 1)
        for i in range(0, n, step)[:max_slices_per_series]:
            arr = window_image(load_slice_array(series.slice_paths[i]))
            vectors.append(embed_image_array(arr))
    if not vectors:
        raise ValueError("No slices found to embed for this study.")
    stacked = np.stack(vectors)
    mean_vec = stacked.mean(axis=0)
    return mean_vec / (np.linalg.norm(mean_vec) + 1e-8)


def build_index_for_studies(study_uids: list[str], data_root: str | None = None) -> dict[str, np.ndarray]:
    """Explicit, opt-in batch embedding — pass a controlled study list (e.g.
    the pilot batch) rather than defaulting to all 4407 studies."""
    from viewer.dicom_utils import find_series_for_study, DEFAULT_DATA_ROOT

    root = data_root or DEFAULT_DATA_ROOT
    embeddings = {}
    for uid in study_uids:
        try:
            series_list = find_series_for_study(uid, data_root=root)
            embeddings[uid] = embed_study(series_list)
        except Exception as exc:
            print(f"[skip] {uid}: {exc}")
    return embeddings
