"""
On-demand, cached mri-interpreter report generation for a single study.

Design (per Dr. Sandeep's decision): generate a study's AI report the FIRST time
it's opened in the viewer, then cache it to disk keyed by StudyInstanceUID so
re-opening the same study never re-spends an LLM call. This module deliberately
does NOT pre-generate reports for all 4407 studies.

Requires an Anthropic API key. On Kaggle, add it as a Kaggle Secret named
ANTHROPIC_API_KEY and enable internet access on the notebook; locally, set the
ANTHROPIC_API_KEY environment variable.

Usage:
    from ai_report import get_or_generate_report
    report_text = get_or_generate_report(study_uid, slice_png_paths, plane_labels)
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Iterable

CACHE_DIR = Path(os.environ.get("AI_REPORT_CACHE_DIR", "reports_cache"))
PROMPT_PATH = Path(__file__).parent / "mri_interpreter_prompt.md"
MODEL = os.environ.get("AI_REPORT_MODEL", "claude-sonnet-4-5")

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
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    # Kaggle Secrets fallback (only works inside a running Kaggle kernel)
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No ANTHROPIC_API_KEY found. Set it as an environment variable, "
            "or add it as a Kaggle Secret named ANTHROPIC_API_KEY and enable "
            "internet on this notebook."
        ) from exc


def _encode_image(path: str) -> dict:
    data = Path(path).read_bytes()
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(data).decode("utf-8"),
        },
    }


def generate_report(
    study_uid: str,
    slice_png_paths: Iterable[str],
    plane_labels: Iterable[str] | None = None,
    max_images: int = 12,
) -> str:
    """Call the Anthropic API once to generate a structured report for this study.

    slice_png_paths: representative rendered slices (not necessarily the full
    series — pass a spread across the sagittal/axial/coronal runs available,
    e.g. every 3rd slice, to stay within a reasonable image budget per call).
    """
    import anthropic

    client = anthropic.Anthropic(api_key=_get_api_key())
    paths = list(slice_png_paths)[:max_images]
    labels = list(plane_labels) if plane_labels else [Path(p).stem for p in paths]

    content = []
    for path, label in zip(paths, labels):
        content.append({"type": "text", "text": f"Slice: {label}"})
        content.append(_encode_image(path))
    content.append(
        {
            "type": "text",
            "text": (
                f"Study {study_uid}. {len(paths)} representative slices shown "
                f"above out of the full series. Generate the structured report "
                f"per the system instructions."
            ),
        }
    )

    resp = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=_load_system_prompt(),
        messages=[{"role": "user", "content": content}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


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
