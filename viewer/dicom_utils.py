"""
DICOM loading, windowing, and study/series indexing for the viewer.

Designed to run against the competition data as mounted by Kaggle at
/kaggle/input/rsna-knee-abnormality-detection/train_series/<StudyInstanceUID>/
<SeriesInstanceUID>/<SOPInstanceUID>.dcm — i.e. no local copy of the 4407-study
corpus is required; this reads directly from Kaggle's own mount.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pydicom

def _detect_data_root() -> str:
    """Kaggle has mounted a competition's data at different paths across
    platform versions -- directly under /kaggle/input/<slug>/, or nested
    under /kaggle/input/competitions/<slug>/. Check both rather than assuming
    one, so this doesn't silently break again on the next mount convention
    change; falls back to the older top-level path if neither is found (the
    notebook's own pilot-batch fallback then takes over)."""
    candidates = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


DEFAULT_DATA_ROOT = os.environ.get("RSNA_DATA_ROOT") or _detect_data_root()


@dataclass
class SeriesInfo:
    study_uid: str
    series_uid: str
    plane: str
    sequence_desc: str
    slice_paths: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Order slices by InstanceNumber for correct scroll order.
        def inst_num(p):
            try:
                return int(pydicom.dcmread(p, stop_before_pixels=True).InstanceNumber)
            except Exception:
                return 0

        self.slice_paths = sorted(self.slice_paths, key=inst_num)


def find_series_for_study(study_uid: str, data_root: str = DEFAULT_DATA_ROOT) -> list[SeriesInfo]:
    """Walk the Kaggle-mounted train_series/<study>/ folder and group by series."""
    study_dir = Path(data_root) / "train_series" / study_uid
    if not study_dir.exists():
        raise FileNotFoundError(
            f"{study_dir} not found — check RSNA_DATA_ROOT / that this study is "
            f"attached as a Kaggle competition data source."
        )

    series_map: dict[str, list[str]] = {}
    for series_dir in study_dir.iterdir():
        if not series_dir.is_dir():
            continue
        slices = [str(p) for p in series_dir.glob("*.dcm")]
        if slices:
            series_map[series_dir.name] = slices

    out = []
    for series_uid, slices in series_map.items():
        sample = pydicom.dcmread(slices[0], stop_before_pixels=True)
        desc = getattr(sample, "SeriesDescription", "")
        plane = (
            "Sagittal" if "SAG" in desc.upper()
            else "Axial" if "AX" in desc.upper()
            else "Coronal" if "COR" in desc.upper()
            else "?"
        )
        out.append(
            SeriesInfo(
                study_uid=study_uid,
                series_uid=series_uid,
                plane=plane,
                sequence_desc=desc,
                slice_paths=slices,
            )
        )
    return sorted(out, key=lambda s: s.plane)


@functools.lru_cache(maxsize=512)
def load_slice_array(path: str) -> np.ndarray:
    """Load one DICOM slice as a rescaled float32 array (cached — repeat scrubs are free)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept


def window_image(arr: np.ndarray, center: float | None = None, width: float | None = None) -> np.ndarray:
    """Apply window/level, defaulting to a robust percentile window (standard
    DICOM-viewer windowing; pass explicit center/width for manual adjustment)."""
    if center is None or width is None:
        lo, hi = np.percentile(arr, 0.5), np.percentile(arr, 99.5)
    else:
        lo, hi = center - width / 2, center + width / 2
    clipped = np.clip(arr, lo, hi)
    return (clipped - lo) / (hi - lo + 1e-6)


def slice_metadata(path: str) -> dict:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return {
        "SeriesDescription": getattr(ds, "SeriesDescription", ""),
        "InstanceNumber": getattr(ds, "InstanceNumber", None),
        "SliceThickness": getattr(ds, "SliceThickness", None),
        "PixelSpacing": getattr(ds, "PixelSpacing", None),
        "Manufacturer": getattr(ds, "Manufacturer", ""),
        "MagneticFieldStrength": getattr(ds, "MagneticFieldStrength", None),
        "RepetitionTime": getattr(ds, "RepetitionTime", None),
        "EchoTime": getattr(ds, "EchoTime", None),
        "Laterality": getattr(ds, "Laterality", "") or getattr(ds, "ImageLaterality", ""),
    }
