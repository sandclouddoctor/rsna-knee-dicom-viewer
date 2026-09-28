"""
On-demand, cached mri-interpreter report generation for a single study.

Design (per Dr. Sandeep's decision): generate a study's AI report the FIRST time
it's opened in the viewer, then cache it to disk keyed by StudyInstanceUID so
re-opening the same study never re-spends an LLM call. This module deliberately
does NOT pre-generate reports for all 4407 studies.

Requires a Gemini API key. On Kaggle, add it as a Kaggle Secret named
GEMINI_API_KEY and enable internet access on the notebook; locally, set the
GEMINI_API_KEY environment variable. Get one at https://aistudio.google.com/apikey
(a free tier is available, which is why this module uses Gemini rather than a
paid-only provider).

Usage:
    from ai_report import get_or_generate_report
    report_text = get_or_generate_report(study_uid, slice_png_paths, plane_labels)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

CACHE_DIR = Path(os.environ.get("AI_REPORT_CACHE_DIR", "reports_cache"))
PROMPT_PATH = Path(__file__).parent / "mri_interpreter_prompt.md"
MODEL = os.environ.get("AI_REPORT_MODEL", "gemini-2.5-flash")

_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


def _cache_path(study_uid: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{study_uid}.json"


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    # Kaggle Secrets fallback (only works inside a running Kaggle kernel)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("GEMINI_API_KEY")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No GEMINI_API_KEY found. Set it as an environment variable, or "
            "add it as a Kaggle Secret named GEMINI_API_KEY and enable "
            "internet on this notebook. Get a key at "
            "https://aistudio.google.com/apikey."
        ) from exc


def generate_report(
    study_uid: str,
    slice_png_paths: Iterable[str],
    plane_labels: Iterable[str] | None = None,
    max_images: int = 12,
) -> str:
    """Call the Gemini API once to generate a structured report for this study.

    slice_png_paths: representative rendered slices (not necessarily the full
    series — pass a spread across the sagittal/axial/coronal runs available,
    e.g. every 3rd slice, to stay within a reasonable image budget per call).
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=_get_api_key())
    paths = list(slice_png_paths)[:max_images]
    labels = list(plane_labels) if plane_labels else [Path(p).stem for p in paths]

    contents = []
    for path, label in zip(paths, labels):
        contents.append(f"Slice: {label}")
        contents.append(
            types.Part.from_bytes(data=Path(path).read_bytes(), mime_type="image/png")
        )
    contents.append(
        f"Study {study_uid}. {len(paths)} representative slices shown above out "
        f"of the full series. Generate the structured report per the system "
        f"instructions."
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=_load_system_prompt(),
            max_output_tokens=2000,
        ),
    )
    return resp.text


def get_or_generate_report(
    study_uid: str,
    slice_png_paths: Iterable[str],
    plane_labels: Iterable[str] | None = None,
    force_regenerate: bool = False,
) -> str:
    """Cached wrapper — the function the viewer notebook should call."""
    cache_file = _cache_path(study_uid)
    if cache_file.exists() and not force_regenerate:
        return json.loads(cache_file.read_text(encoding="utf-8"))["report"]

    report = generate_report(study_uid, slice_png_paths, plane_labels)
    cache_file.write_text(
        json.dumps({"study_uid": study_uid, "report": report}, indent=2),
        encoding="utf-8",
    )
    return report
