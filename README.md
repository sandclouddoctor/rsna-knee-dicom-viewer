# RSNA Knee DICOM Viewer + DINOv3 Multimodal RAG

Custom tooling for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (4407 training studies, 12 abnormality labels, paired
radiology reports).

Two components, built together but usable independently:

1. **`viewer/`** — a DICOM viewer notebook that runs directly against Kaggle's
   own mount of the competition data (no bulk download of the ~150GB+ raw
   corpus). Beyond standard viewer features (series/slice navigation,
   windowing, zoom), it adds:
   - 12 radio-button controls for the competition's labels (ACL, MCL, Medial
     Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
     Synovitis, Baker's, Contusion, Fracture), pre-filled from ground truth
     where available
   - the original-language competition report plus on-demand English
     translation
   - an on-demand, cached `/mri-interpreter`-style AI structured report per
     study
2. **`rag/`** — a DINOv3 multimodal RAG: image embeddings for case-similarity
   retrieval, a text index over the `mri-interpreter` knee knowledge base, and
   a retrieval-augmented classifier sketch that combines both to refine an
   existing model's uncertain predictions.

## Status: pilot, not full-scale

This was deliberately built and validated against a **pilot batch of 24
studies** (`data/pilot_studies.csv`, spanning 0-9 positive findings per case)
rather than all 4407 — see `docs/ARCHITECTURE.md` for why, and for the
concrete steps to scale up once the design is validated.

## Setup

**On Kaggle** (recommended — this is where the data lives):
1. Fork/open `viewer/dicom_viewer.ipynb` as a Kaggle notebook with the
   competition attached as a data source.
2. Settings → Internet → On (needed for translation + AI report generation).
3. Add-ons → Secrets → add `ANTHROPIC_API_KEY` if you want AI report
   generation.
4. `pip install -r requirements.txt` in the first cell, then run all.

**Locally** (for development against the pilot batch only — you won't have
`/kaggle/input` mounted, so the notebook falls back to `data/pilot_studies.csv`
and you'll need those 24 studies' DICOMs downloaded separately):
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...       # for AI report generation
export RSNA_DATA_ROOT=/path/to/your/local/copy
jupyter notebook viewer/dicom_viewer.ipynb
```

## Repo layout

```
viewer/
  dicom_viewer.ipynb       # main viewer notebook (ipywidgets)
  dicom_utils.py           # series discovery, slice loading, windowing
  translate.py             # on-demand cached report translation
  ai_report.py             # on-demand cached mri-interpreter report generation
  mri_interpreter_prompt.md  # the system prompt ai_report.py sends
rag/
  knowledge/knee_mri_knowledge.md  # source knowledge, chunked by ## heading
  knowledge_base.py         # chunking + text embedding
  dinov3_embed.py           # DINOv3 study embedding (opt-in, not all-4407-by-default)
  vector_store.py           # FAISS wrappers (image index + text index)
  retrieve.py               # combined retrieval entry point
  classifier.py             # retrieval-augmented classifier sketch
data/
  pilot_studies.csv         # 24-study pilot batch (StudyInstanceUID + 12 labels)
  pilot_reports.csv         # original-language reports for the same 24 studies
  labels_schema.json        # the 12 label names/encoding
docs/
  ARCHITECTURE.md           # design decisions and scale-up plan
```

## Cost/quota notes

- AI report generation and translation are **on-demand and cached to disk**
  per study — opening the same study twice never re-spends an API call.
- DINOv3 embedding is **opt-in per study list** (`build_index_for_studies`
  takes an explicit list) — it will not silently embed all 4407 studies.
- None of this repo's code runs Kaggle/Colab GPU training jobs; it's read/
  inference tooling layered on top of your existing Cycle4 model work.
