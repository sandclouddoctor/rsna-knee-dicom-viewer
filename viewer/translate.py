"""
English translation of the competition's original-language radiology reports
(train.csv 'Report' column -- Spanish, Dutch, German, etc.).

Primary source: a bundled, pre-computed translation covering all 4407 training
studies (data/train_with_english_translation.csv, StudyInstanceUID ->
report-english-translation). This is free and instant -- no API call, no rate
limit, no internet dependency -- and covers every study in the competition, so
it is checked first and handles the overwhelming majority of lookups.

Fallback only: if a study isn't in the bundled file (e.g. it's outside the
original 4407, or the bundle isn't present in this environment), falls back to
an on-demand, disk-cached translation via deep-translator's GoogleTranslator
backend (needs internet). Falls back further to returning the original text
with a warning if that also fails, rather than crashing the viewer.

Usage:
    from translate import get_or_translate
    english_text = get_or_translate(study_uid, original_report_text)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(os.environ.get("TRANSLATION_CACHE_DIR", "translation_cache"))

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BUNDLED_CANDIDATES = [
    _REPO_ROOT / "data" / "train_with_english_translation.csv",
    Path("data/train_with_english_translation.csv"),
]

_bundled_lookup: Optional[dict] = None  # lazy-loaded, module-level cache


def _load_bundled_lookup() -> dict:
    global _bundled_lookup
    if _bundled_lookup is not None:
        return _bundled_lookup

    _bundled_lookup = {}
    for path in _BUNDLED_CANDIDATES:
        if not path.exists():
            continue
        try:
            import pandas as pd

            df = pd.read_csv(path, usecols=["StudyInstanceUID", "report-english-translation"])
            _bundled_lookup = dict(
                zip(df["StudyInstanceUID"].astype(str), df["report-english-translation"])
            )
        except Exception as exc:  # malformed/partial bundle -- degrade to the API fallback
            print(f"Warning: could not load bundled translations from {path}: {exc}")
            _bundled_lookup = {}
        break
    return _bundled_lookup


def _cache_path(study_uid: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{study_uid}.json"


def translate_to_english(text: str) -> str:
    if not text or not text.strip():
        return text
    from deep_translator import GoogleTranslator

    # GoogleTranslator has a ~5000 char limit per call; radiology reports are
    # short enough in practice, but chunk defensively just in case.
    chunks = [text[i : i + 4500] for i in range(0, len(text), 4500)]
    translated_chunks = [
        GoogleTranslator(source="auto", target="en").translate(chunk)
        for chunk in chunks
    ]
    return " ".join(translated_chunks)


def get_or_translate(study_uid: str, original_text: str, force_retranslate: bool = False) -> str:
    # 1. Bundled pre-computed translation: free, instant, covers all 4407
    #    studies. This is what almost every call resolves to.
    if not force_retranslate:
        bundled = _load_bundled_lookup().get(str(study_uid))
        if bundled is not None and str(bundled).strip():
            return bundled

    # 2. Disk cache from a prior live translation (covers studies outside the
    #    bundle, or an explicit force_retranslate request).
    cache_file = _cache_path(study_uid)
    if cache_file.exists() and not force_retranslate:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("original") == original_text:
            return cached["translated"]

    # 3. Live API call, as a last resort.
    try:
        translated = translate_to_english(original_text)
    except Exception as exc:  # network/API issues -- degrade gracefully
        translated = f"[translation unavailable: {exc}]\n\nOriginal:\n{original_text}"

    cache_file.write_text(
        json.dumps(
            {"study_uid": study_uid, "original": original_text, "translated": translated},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return translated
