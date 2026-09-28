# RSNA Knee DICOM Viewer + DINOv3 Model-Ensemble RAG

Custom tooling for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
Kaggle competition (4407 training studies, 12 abnormality labels, paired
radiology reports).

Two components. They are **deliberately kept separate** — different purpose,
different consumer, no code dependency in either direction:

1. **`viewer/`** — Dr. Sandeep's own clinician annotation tool. A DICOM viewer
   notebook that runs directly against Kaggle's own mount of the competition
   data (no bulk download of the ~150GB+ raw corpus), used to build ground
   truth on the 4407 studies by hand. Beyond standard viewer features
   (series/slice navigation, windowing, zoom), it adds:
   - 12 radio-button controls for the competition's labels (ACL, MCL, Medial
     Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion,
     Synovitis, Baker's, Contusion, Fracture) — pre-filled with your own saved
     label where you've already annotated a study, else the competition's
     ground truth where available
   - a **"Save my labels" button** that writes your labels to
     `data/clinician_annotations.csv`, keyed by study + annotator, so your
     work persists across sessions and is reloaded automatically next time
     you open that study
   - the original-language competition report plus its English translation —
     shown automatically, since it's pre-computed and bundled for all 4407
     studies (`data/train_with_english_translation.csv`), not a live API call
   - an on-demand, cached `/mri-interpreter`-style AI structured report per
     study, offered as a second opinion to speed up your read — not a
     replacement for your own label
2. **`model_ensemble/`** — the competition modeling component: a DINOv3
   multimodal RAG (image embeddings for case-similarity retrieval, a text
   index over the `mri-interpreter` knee knowledge base) plus a
   retrieval-augmented classifier that refines an existing model's uncertain
   predictions. This is for the model ensemble pipeline, not the viewer —
   nothing in `viewer/` imports it, and nothing in it imports `viewer/`.

## Status: pilot, not full-scale

This was deliberately built and validated against a **pilot batch of 24
studies** (`data/pilot_studies.csv`, spanning 0-9 positive findings per case)
rather than all 4407 — see `docs/ARCHITECTURE.md` for why, and for the
concrete steps to scale up once the design is validated.

## Setup

**On Kaggle** (recommended — this is where the data lives):
1. Fork/open `viewer/dicom_viewer.ipynb` as a Kaggle notebook with the
   competition attached as a data source.
2. Settings → Internet → On (needed for AI report generation; translation is
   bundled and works offline).
3. Add-ons → Secrets → add `GEMINI_API_KEY` if you want AI report generation
   (a free tier is available at https://aistudio.google.com/apikey).
4. `pip install -r requirements.txt` in the first cell, then run all.
5. Save your labels as you go with **Save my labels** — they land in
   `data/clinician_annotations.csv`. Save that file as a Kaggle Dataset (or
   attach it as notebook output) if you want your annotations to persist
   across kernel sessions/versions.

**Locally** (for development against the pilot batch only — you won't have
`/kaggle/input` mounted, so the notebook falls back to `data/pilot_studies.csv`
and you'll need those 24 studies' DICOMs downloaded separately):
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...          # for AI report generation
export RSNA_DATA_ROOT=/path/to/your/local/copy
jupyter notebook viewer/dicom_viewer.ipynb
```

## Repo layout

```
viewer/                          # CLINICIAN ANNOTATION TOOL (separate from model_ensemble/)
  dicom_viewer.ipynb             # main viewer/labeling notebook (ipywidgets)
  dicom_utils.py                 # series discovery, slice loading, windowing
  annotations.py                 # save/load your labels -> data/clinician_annotations.csv
  translate.py                   # on-demand cached report translation
  ai_report.py                   # on-demand cached mri-interpreter report generation
  mri_interpreter_prompt.md      # the system prompt ai_report.py sends
model_ensemble/                  # COMPETITION MODELING COMPONENT (separate from viewer/)
  knowledge/knee_mri_knowledge.md  # source knowledge, chunked by ## heading
  knowledge_base.py              # chunking + text embedding
  dinov3_embed.py                # DINOv3 study embedding (opt-in, not all-4407-by-default)
  vector_store.py                # FAISS wrappers (image index + text index)
  retrieve.py                    # combined retrieval entry point
  classifier.py                  # retrieval-augmented classifier for the ensemble
data/
  pilot_studies.csv              # 24-study pilot batch (StudyInstanceUID + 12 labels)
  pilot_reports.csv              # original-language reports for the same 24 studies
  train_with_english_translation.csv  # pre-computed English translation for all 4407 studies
  clinician_annotations.csv      # your saved labels (created the first time you click Save)
  labels_schema.json             # the 12 label names/encoding
docs/
  ARCHITECTURE.md                # design decisions and scale-up plan
```

## Cost/quota notes

- English translation is **free and instant for all 4407 studies** — it's a
  pre-computed lookup (`data/train_with_english_translation.csv`), not a live
  API call. `translate.py` only falls back to a live, disk-cached
  Google-Translate call for a study outside that bundle, or if you click
  "Re-translate."
- AI report generation is **on-demand and cached to disk** per study —
  opening the same study twice never re-spends an API call.
- DINOv3 embedding is **opt-in per study list** (`build_index_for_studies`
  takes an explicit list) — it will not silently embed all 4407 studies.
- None of this repo's code runs Kaggle/Colab GPU training jobs; it's read/
  inference tooling layered on top of your existing Cycle4 model work.
