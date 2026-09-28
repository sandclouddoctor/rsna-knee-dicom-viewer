"""
Cross-session progress persistence for the DICOM viewer.

Kaggle interactive sessions get recycled (idle timeout, ~9h hard cap) --
labeling 4407 studies over many days means each new "Edit" session starts
from a clean /kaggle/working, with no memory of which study you were on or
what you'd saved so far, unless that state is pushed somewhere durable
in between.

This module pushes data/clinician_annotations.csv and a small progress.json
(which study you last had open) to a companion Kaggle Dataset
(docdsandeep/rsna-knee-viewer-progress) whenever you click "Save progress",
and restores from that dataset's mounted copy at the top of every session --
so reopening the notebook days later picks up exactly where you left off.

Requires KAGGLE_USERNAME and KAGGLE_KEY, e.g. from Kaggle Secrets (the same
values already used for the GitHub Actions deploy workflow), set as
environment variables before sync_progress() is called. This module never
reads secrets itself -- the notebook's setup cell does that once at start,
same as it does for GEMINI_API_KEY.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

PROGRESS_DATASET_SLUG = "docdsandeep/rsna-knee-viewer-progress"
PROGRESS_MOUNT = Path("/kaggle/input/rsna-knee-viewer-progress")
WORKING_DATA_DIR = Path(os.getcwd()) / "data"
PROGRESS_FILE = WORKING_DATA_DIR / "progress.json"


def restore_from_last_session() -> dict:
    """Copy the last-synced annotations CSV + progress pointer from the
    mounted progress dataset into the writable working dir, if present.

    Returns the restored progress dict, e.g. {"last_study_uid": "...",
    "updated_at": "..."}, or {} on the very first-ever session (nothing
    synced yet, or the dataset isn't attached to this kernel).
    """
    WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PROGRESS_MOUNT.is_dir():
        return {}

    src_csv = PROGRESS_MOUNT / "clinician_annotations.csv"
    if src_csv.exists():
        shutil.copy(src_csv, WORKING_DATA_DIR / "clinician_annotations.csv")

    src_progress = PROGRESS_MOUNT / "progress.json"
    if src_progress.exists():
        try:
            return json.loads(src_progress.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def sync_progress(
    current_study_uid: str,
    annotations_path: str,
    status_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """Push the current study pointer + annotations CSV as a new version of
    the progress Dataset. Call this from the "Save progress" button.

    Returns a short human-readable result string (also passed to status_cb,
    if given, so a caller can stream "Uploading..." -> final result to a
    notebook Output widget rather than just getting it at the end).
    """

    def _status(msg: str) -> None:
        if status_cb:
            status_cb(msg)

    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        return (
            "No KAGGLE_USERNAME/KAGGLE_KEY set -- add both as Kaggle Secrets on "
            "this notebook (same values used for the GitHub Actions deploy "
            "workflow) to enable progress sync."
        )

    WORKING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps({
        "last_study_uid": current_study_uid,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }))

    stage = Path("/kaggle/working/_progress_stage")
    stage.mkdir(parents=True, exist_ok=True)
    if os.path.exists(annotations_path):
        shutil.copy(annotations_path, stage / "clinician_annotations.csv")
    else:
        # Nothing saved yet -- still sync the position pointer with an empty
        # placeholder so the dataset always has both files in a known shape.
        (stage / "clinician_annotations.csv").write_text("")
    shutil.copy(PROGRESS_FILE, stage / "progress.json")
    (stage / "dataset-metadata.json").write_text(json.dumps({
        "title": "RSNA Knee Viewer Progress",
        "id": PROGRESS_DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}],
    }))

    _status("Uploading progress to Kaggle...")
    result = subprocess.run(
        ["kaggle", "datasets", "version", "-p", str(stage),
         "-m", f"Progress sync {datetime.now(timezone.utc).isoformat()}",
         "-r", "tar"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        msg = f"Save failed: {result.stderr.strip() or result.stdout.strip()}"
        _status(msg)
        return msg

    msg = f"Progress saved (study {current_study_uid}) -- safe to close this session."
    _status(msg)
    return msg
