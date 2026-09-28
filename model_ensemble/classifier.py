"""
Retrieval-augmented classifier for the 12 RSNA labels — competition
model-ensemble component.

DELIBERATELY SEPARATE from viewer/: nothing in viewer/ imports this module,
and nothing here imports viewer/. This lives entirely on the modeling side —
an ensemble member alongside your Cycle4 DINOv3 arm6/arm7 models, consumed by
your training/inference pipeline, not by the clinician-facing DICOM viewer.

It combines:
  1. a k-NN prior over the 12 labels from similar prior cases (by DINOv3
     embedding distance, via model_ensemble.retrieve.retrieve_similar_cases —
     the case index's ground truth comes from data/clinician_annotations.csv
     plus the original manually-verified subset, i.e. from what the viewer
     produces, without this module depending on the viewer's code), and
  2. relevant mri-interpreter knowledge chunks for the findings your primary
     model flags as uncertain,
into a single LLM call that returns calibrated 12-label predictions plus a
short rationale citing which retrieved cases/knowledge informed each call.

Intended use: run your Cycle4 model first to get its per-label probabilities,
then call refine_predictions() only for labels near the decision boundary
(e.g. 0.3-0.7) where retrieval context is most likely to help — not as a
blanket replacement for the trained model's output on every label.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]


def knn_label_prior(similar_cases: List[dict]) -> Dict[str, float]:
    """Simple distance-weighted vote over the 12 labels from retrieved similar
    cases' known ground truth (only meaningful for cases in the ~58-study
    manually-labeled subset — filter similar_cases to those before calling)."""
    weighted_sum = {lbl: 0.0 for lbl in LABELS}
    weight_total = 0.0
    for case in similar_cases:
        labels = case.get("metadata", {}).get("labels", {})
        if not labels:
            continue
        weight = max(case["score"], 0.0)
        weight_total += weight
        for lbl in LABELS:
            val = labels.get(lbl)
            if val is not None:
                weighted_sum[lbl] += weight * float(val)
    if weight_total == 0:
        return {lbl: None for lbl in LABELS}
    return {lbl: weighted_sum[lbl] / weight_total for lbl in LABELS}


def refine_predictions(
    study_uid: str,
    model_probs: Dict[str, float],
    similar_cases: List[dict],
    uncertain_labels: List[str] | None = None,
    low: float = 0.3,
    high: float = 0.7,
) -> dict:
    """Call an LLM with the primary model's probabilities, the k-NN prior from
    similar cases, and retrieved knowledge snippets for the uncertain labels,
    and ask it to return a refined judgement + rationale. Returns the raw
    text response — parse/structure further as your pipeline needs.
    """
    import anthropic
    from .retrieve import retrieve_knowledge

    uncertain = uncertain_labels or [
        lbl for lbl, p in model_probs.items() if low <= p <= high
    ]
    if not uncertain:
        return {"study_uid": study_uid, "refined": False, "reason": "no uncertain labels", "model_probs": model_probs}

    prior = knn_label_prior(similar_cases)
    knowledge_snippets = {}
    for lbl in uncertain:
        hits = retrieve_knowledge(f"{lbl} knee MRI grading and pitfalls", k=2)
        knowledge_snippets[lbl] = [h["metadata"]["text"] for h in hits]

    prompt = {
        "study_uid": study_uid,
        "model_probabilities": model_probs,
        "uncertain_labels": uncertain,
        "knn_prior_from_similar_cases": {lbl: prior[lbl] for lbl in uncertain},
        "similar_cases_summary": [
            {"study_uid": c["id"], "similarity": c["score"], "labels": c.get("metadata", {}).get("labels")}
            for c in similar_cases[:5]
        ],
        "relevant_knowledge": knowledge_snippets,
    }

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=os.environ.get("AI_REPORT_MODEL", "claude-sonnet-4-5"),
        max_tokens=1000,
        system=(
            "You are refining an ML model's uncertain knee-MRI abnormality "
            "predictions using retrieved similar cases and radiology knowledge. "
            "For each uncertain label, weigh the model's own probability against "
            "the k-NN prior from similar cases (note: prior is often None if no "
            "similar case has ground truth — treat that as no signal, not "
            "evidence of absence) and the knowledge snippet's grading criteria. "
            "Return JSON: {label: {refined_probability, rationale}} for each "
            "uncertain label only. Be conservative — don't override the model's "
            "probability by a large margin without a clear reason from the "
            "retrieved context."
        ),
        messages=[{"role": "user", "content": json.dumps(prompt, indent=2)}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return {"study_uid": study_uid, "refined": True, "raw_response": text, "input": prompt}
