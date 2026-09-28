"""
On-demand, cached English translation of the competition's original-language
radiology reports (train.csv 'Report' column — Spanish, Dutch, German, etc.).

Uses deep-translator's GoogleTranslator backend (needs internet — enable it on
the Kaggle notebook, or run locally with normal internet access). Falls back to
returning the original text with a warning if translation fails, rather than
crashing the viewer.

Usage:
    from translate import get_or_translate
    english_text = get_or_translate(study_uid, original_report_text)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("TRANSLATION_CACHE_DIR", "translation_cache"))


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
    cache_file = _cache_path(study_uid)
    if cache_file.exists() and not force_retranslate:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("original") == original_text:
            return cached["translated"]

    try:
        translated = translate_to_english(original_text)
    except Exception as exc:  # network/API issues — degrade gracefully
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
