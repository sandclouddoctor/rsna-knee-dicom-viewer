"""
Package one study's DICOM series + report text into a small zip for local
review in the Claude artifact viewer.

Why this exists: the competition's ~570GB/819k-file corpus can only live on
Kaggle's own mount (bulk-downloading it, or re-hosting it elsewhere, isn't
practical or within the competition's data terms). A single study is small
though -- the corpus averages ~130MB/study -- so the workflow is: export the
ONE study you're about to review from here, download that zip through
Kaggle's own Output panel, then open it in the artifact. Nothing here talks
to model_ensemble/; this is clinician-tool plumbing only.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

EXPORT_DIR = Path(os.getcwd()) / "exports"


def export_study_zip(
    study_uid: str,
    series_list,
    report_original: str,
    report_translated: str,
    ground_truth_labels: dict | None = None,
    out_dir: Path = EXPORT_DIR,
) -> str:
    """Zip one study's DICOM series (raw .dcm files, unmodified) plus its
    report text into out_dir/<study_uid>.zip. Returns the zip's path.

    series_list: the list[SeriesInfo] from dicom_utils.find_series_for_study
    for this study -- reuse what the viewer already loaded rather than
    re-scanning the mount.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{study_uid}.zip"

    manifest = {
        "study_uid": study_uid,
        "series": [
            {
                "index": i,
                "plane": s.plane,
                "sequence_desc": s.sequence_desc,
                "slice_count": len(s.slice_paths),
                "folder": f"series_{i}",
            }
            for i, s in enumerate(series_list)
        ],
    }
    report = {
        "study_uid": study_uid,
        "original": report_original or "",
        "translated": report_translated or "",
        "ground_truth_labels": ground_truth_labels or {},
    }

    # Overwrite any previous export of this study rather than appending to it.
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("report.json", json.dumps(report, indent=2))
        for i, s in enumerate(series_list):
            for slice_path in s.slice_paths:
                arcname = f"series_{i}/{os.path.basename(slice_path)}"
                zf.write(slice_path, arcname)

    return str(zip_path)
