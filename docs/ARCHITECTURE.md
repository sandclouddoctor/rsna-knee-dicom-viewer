# Architecture and design decisions

## Why Kaggle-hosted, not a local/cloud copy

The full training corpus is ~4407 studies, each with 2+ series of 15-40 DICOM
slices — a full download is realistically 100-200GB+. Kaggle already hosts
this data and mounts it read-only into any notebook that attaches the
competition as a data source at `/kaggle/input/rsna-knee-abnormality-detection/`.
The viewer — Dr. Sandeep's clinician annotation tool (`viewer/dicom_utils.py`) —
reads directly from that path, so there is
no bulk transfer step and no local storage burden — you run the notebook on
Kaggle, browse whichever studies you want, and nothing large ever needs to
leave Kaggle's infrastructure.

The tradeoff: the viewer isn't a standalone shareable web app the way a
Streamlit deployment would be — it's a Kaggle notebook you run interactively
(ipywidgets UI renders inline in the notebook). If you later want a
link-shareable app outside Kaggle, the natural next step is a small Streamlit
or FastAPI service that reads from a data source it controls (a subset copied
to a Kaggle Dataset you host, or cloud storage) — worth revisiting once you
know which studies you actually want to share and with whom.

## Why a pilot batch of 24 studies, not all 4407

Two costs scale directly with study count and neither should be paid before
the design is validated:
1. **AI report generation** is an LLM call per study. Pre-generating for 4407
   studies means ~4407 calls before you've confirmed the report format,
   prompt, and caching actually work the way you want.
2. **DINOv3 embedding** for the RAG's case index is a forward pass per study
   (cheap individually, not free at 4407x, and meaningfully more if you later
   embed every slice rather than a representative subset).

Both are designed to be **incremental**: `ai_report.py` caches per-study on
first view (see below), and `dinov3_embed.build_index_for_studies()` takes an
explicit study list rather than defaulting to "all of them." Scaling up is
changing that list, not rewriting the pipeline.

The pilot batch itself (`data/pilot_studies.csv`) was selected from the 58
studies in `train.csv` that have complete manual ground truth across all 12
labels (out of 4407 total — the rest carry only NLP-derived weak/probabilistic
labels from `report_weak_label_manifest.csv`-style scoring, not hard 0/1
truth). It spans 1 to 9+ positive findings per case, plus the demo study used
during development (`...95817260`), so the viewer and RAG can be exercised
against genuinely varied cases rather than only normal or only complex ones.

## Translation: bundled and pre-computed, not on-demand

Unlike AI report generation, English translation is no longer a live API call
in the normal case. `data/train_with_english_translation.csv` bundles a
pre-computed English translation for all 4407 training studies (courtesy of
Dr. Sandeep), keyed by `StudyInstanceUID`. `translate.get_or_translate()`
checks this bundle first — free, instant, no internet dependency, no rate
limit — and only falls back to a live, disk-cached Google-Translate call (the
original on-demand design, described below) for a study outside the bundle,
or when `force_retranslate=True` is passed (wired to the notebook's
"Re-translate" button, for the rare case the bundled translation looks wrong).
This removes the Google-Translate rate-limiting risk this repo hit during its
own smoke-testing (see "Errors and fixes" history) for the entire corpus, not
just the pilot batch.

## On-demand + cached: the AI report lifecycle

`ai_report.get_or_generate_report()` follows the on-demand + cached pattern
that `translate.get_or_translate()` used exclusively before the bundle above
existed, and still falls back to for translation outside the bundle: check a
JSON cache file keyed by `StudyInstanceUID` first; only call the API
(Gemini / Google Translate) on a cache miss; write the result back before
returning. This means:
- Opening a study you've already viewed costs nothing.
- A fresh Kaggle kernel session starts with an empty cache — if you want
  persistence across sessions, save `reports_cache/` (and `translation_cache/`,
  for the fallback path) as a Kaggle Dataset and load it back in at notebook
  start, or attach them as notebook output that persists across versions.
- `force_regenerate=True` (AI reports) / `force_retranslate=True` (translation
  fallback) is available if a report/translation needs to be redone (e.g.
  after editing `mri_interpreter_prompt.md`).

## Viewer vs. model_ensemble: deliberately separate

The viewer (`viewer/`) and the model-ensemble RAG (`model_ensemble/`) are two
independent components with no import in either direction. The viewer is
Dr. Sandeep's clinician annotation tool — its job is producing labels, via
`viewer/annotations.py` writing to `data/clinician_annotations.csv` when you
click "Save my labels" in `dicom_viewer.ipynb`. The model_ensemble RAG is a
competition modeling component consumed by the training/inference pipeline,
not by the viewer. The two connect only at the data layer, and only in one
direction: `data/clinician_annotations.csv` (produced by the viewer) is a
candidate source of additional hard ground truth that a future case-index
build for `model_ensemble/retrieve.py` could draw on, alongside the 58
fully-labeled `train.csv` studies — that wiring is not yet built (see "What
was NOT built here" below).

## The two model_ensemble usage modes

Your brief asked for both, and they share the same underlying indices:

**Knowledge/reference tool** — `model_ensemble.retrieve.retrieve_knowledge(query)`
takes a free-text query (a finding name, a question, a draft impression) and
returns the most relevant chunks of the `mri-interpreter` knee knowledge base
(`model_ensemble/knowledge/knee_mri_knowledge.md`, split by `## ` heading — 12
chunks currently, covering meniscus grading/morphology/mimics, ACL/PCL, MCL/LCL,
posterolateral/posteromedial corners, cartilage grading, bone marrow patterns,
joint fluid/Baker cyst, extensor mechanism, and OA pattern). Useful standalone,
independent of any model — e.g. pull up the ACL assessment criteria while
reviewing a case.

**Retrieval-augmented classifier** — `model_ensemble.classifier.refine_predictions()`
is meant to sit downstream of your existing Cycle4 DINOv3 model, not replace it.
Workflow: run your model to get per-label probabilities → identify labels
near the decision boundary (default 0.3-0.7) → for those labels only, retrieve
(a) similar prior cases by DINOv3 embedding distance
(`model_ensemble.retrieve.retrieve_similar_cases`) and their known labels as a k-NN prior,
and (b) relevant knowledge chunks for that finding → send model probability +
k-NN prior + knowledge snippets to an LLM call that returns a refined
probability and a rationale citing what informed it. This is intentionally
conservative (the prompt tells the model not to override the base model's
probability without clear reason) and intentionally scoped to uncertain labels
only, not a wholesale second opinion on every prediction.

**Important caveat**: the k-NN prior is only informative when a retrieved
"similar case" is itself one of the 58 manually-labeled studies (or has an AI-
generated report you're treating as a soft label) — most of the 4407 studies
have no hard ground truth, so `knn_label_prior()` correctly returns `None` for
labels with no evidence in the retrieved set. Don't feed None into anything
that expects a probability without checking for it first.

## DINOv3 model note

`model_ensemble/dinov3_embed.py` targets `facebook/dinov3-vits16-pretrain-lvd1689m` via
`transformers.AutoModel`, with an automatic fallback to `facebook/dinov2-small`
if DINOv3 weights aren't resolvable in your environment (gated model access,
or a `transformers` version predating DINOv3 support — check on Kaggle before
assuming the fallback fired for the wrong reason). The fallback was exercised
during this repo's own CPU smoke test, since this development environment
didn't have DINOv3 weight access; verify on Kaggle (or wherever you have
proper access) that the real DINOv3 path is what actually runs before trusting
embeddings for anything beyond further smoke-testing.

## What was NOT built here (explicitly out of scope for this pilot)

- No retraining or fine-tuning of your Cycle4 arm6/arm7 models.
- No full-corpus embedding index (4407 studies) — the case index is built
  on-demand from whatever study list you pass in.
- No standalone hosted web app outside Kaggle/local notebook use.
- No automatic full-dataset AI report pre-generation.
- No pipeline wiring `data/clinician_annotations.csv` (what the viewer
  produces) into `model_ensemble/retrieve.py`'s case index — the viewer can
  produce ground truth today, but nothing yet consumes it on the modeling
  side. That's the natural next step once you've built up enough annotations
  to be worth indexing.

Each of these is a reasonable next step once you've used the pilot and want to
scale a specific piece — flag which one and it's a much narrower, well-scoped
follow-up rather than a from-scratch build.
