"""
Clinician label persistence for the DICOM viewer.

This is the ground-truth-capture mechanism the viewer was actually built for:
Dr. Sandeep reviews a study in viewer/dicom_viewer.ipynb, sets/corrects the 12
RadioButtons, and clicks "Save my labels" — that writes a row here. Nothing in
this module talks to model_ensemble/ (the competition retrieval-augmented
classifier); it is consumed by that side only indirectly, by whoever later
builds a case index from data/clinician_annotations.csv as an additional
source of hard ground truth beyond the 58 fully-labeled train.csv studies.

Storage: a single flat CSV, one row per (StudyInstanceUID, annotator) pair —
re-saving the same study by the same annotator overwrites their previous row
in place rather than appending a duplicate, so the file is always "latest
label per annotator per study," not an audit log.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

ANNOTATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "clinician_annotations.csv",
)

_COLUMNS = ["StudyInstanceUID", *LABELS, "annotator", "timestamp", "notes"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLUMNS)


def load_annotations(path: str = ANNOTATIONS_PATH) -> pd.DataFrame:
    """All saved clinician annotations, one row per (study, annotator)."""
    if not os.path.exists(path):
        return _empty_df()
    df = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    for col in _COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[_COLUMNS]


def get_labels_for_study(
    study_uid: str, annotator: Optional[str] = None, path: str = ANNOTATIONS_PATH
) -> Optional[Dict[str, int]]:
    """Most recent saved labels for a study, or None if never annotated.

    If annotator is given, only that annotator's row is considered; otherwise
    the most recently-saved row for the study (any annotator) is returned.
    """
    df = load_annotations(path)
    if df.empty:
        return None
    rows = df[df["StudyInstanceUID"] == str(study_uid)]
    if annotator is not None:
        rows = rows[rows["annotator"] == annotator]
    if rows.empty:
        return None
    row = rows.sort_values("timestamp").iloc[-1]
    return {lbl: (None if pd.isna(row[lbl]) else int(row[lbl])) for lbl in LABELS}


def save_annotation(
    study_uid: str,
    labels: Dict[str, int],
    annotator: str = "Dr. Sandeep",
    notes: str = "",
    path: str = ANNOTATIONS_PATH,
) -> None:
    """Save/overwrite this annotator's labels for a study.

    labels: dict mapping label name -> 0, 1, or -1/None for "left unlabeled"
    (a -1/None value is stored as blank, matching train.csv's own convention
    for "not manually assessed," so it never gets mistaken for a real 0).
    """
    unknown = set(labels) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown label(s): {sorted(unknown)}")

    df = load_annotations(path)
    mask = (df["StudyInstanceUID"] == str(study_uid)) & (df["annotator"] == annotator)
    df = df[~mask]

    row = {"StudyInstanceUID": str(study_uid), "annotator": annotator,
           "timestamp": datetime.now(timezone.utc).isoformat(), "notes": notes}
    for lbl in LABELS:
        val = labels.get(lbl, None)
        row[lbl] = pd.NA if val is None or val < 0 else int(val)

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)[_COLUMNS]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def annotation_progress(study_uids, path: str = ANNOTATIONS_PATH) -> Dict[str, int]:
    """Quick coverage summary: how many of study_uids have a saved clinician
    annotation vs. still need review. Useful for tracking labeling progress
    across a batch."""
    df = load_annotations(path)
    annotated = set(df["StudyInstanceUID"]) if not df.empty else set()
    total = len(list(study_uids))
    done = sum(1 for u in study_uids if str(u) in annotated)
    return {"total": total, "annotated": done, "remaining": total - done}
