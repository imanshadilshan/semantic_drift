# Progress

## Status: Day 1-3 (repo scaffold + literature review) mostly done; design finalization in progress

## Done
- Repo scaffolding: folder structure under `Implementation/` (src, tests, data, configs, notebooks, results, human_eval).
- `requirements.txt`, `configs/default.yaml`, `data/edit_instructions.json` (schema example) in place.
- All `src/` modules stubbed with docstrings and `raise NotImplementedError` bodies.
- `src/utils.py` has working `get_device()` / `set_seed()`.
- `.gitignore` set up to keep `raw_images/`, `results/`, and model checkpoints out of git.
- Literature review written: `../Literature Review/literature_review.md`, based on the 12 downloaded PDFs in `../Research Papers - Existing/`. Key finding: MagicBrush (NeurIPS 2023) already has multi-turn edit data — good validation of the Chain A/B design, but its evaluation doesn't isolate collateral/non-target-region change, which is still the gap this project fills. MasaCtrl and Plug-and-Play are the closest prior art to the masked-conditioning/region-locking mitigations.

- **Day 4-5 — Drift Score logic**: `src/drift_score.py` implemented (`cosine_similarity()`, `compute_drift_score()`, `compute_chain_drift_score()`). `compute_chain_drift_score()` sums per-step scores (cumulative drift across a chain). Both functions raise `ValueError` on empty input (all regions were targets / empty chain) rather than silently returning 0 or NaN. `tests/test_drift_score.py` has 10 tests using dummy vectors as region crops and an identity `embed_fn` — no real model or image involved. All 10 pass (`python -m pytest tests/test_drift_score.py -v`).

- **Day 6-9 — Dataset download**: `scripts/download_coco_subset.py` written and run. Downloads COCO val2017 annotations (cached in `data/.cache/`, gitignored), filters images to those with 2-8 annotated objects (enough for target + non-target regions without over-cluttering), randomly samples 60 (seed 42, reproducible), downloads the actual jpgs into `data/raw_images/`, and writes `data/coco_subset_manifest.json` recording each image's COCO category labels. 60/60 images downloaded successfully.

- **Day 6-9 — Edit instructions**: `data/edit_instructions.json` fully populated — every one of the 60 images actually viewed (via Read, not inferred from category labels alone) and given a custom 4-step Chain A (object-level) and 4-step Chain B (global), 120 chains total. Verified against the manifest: 0 missing images, 60 object_level + 60 global, all chains length 4.

- **Day 10-13 — Model wrappers**: `src/edit_runner.py` (InstructPix2Pix, `diffusers` pipeline) and `src/segment.py` (SAM, `segment_anything` package) implemented — both lazy-load their models on first call so importing them doesn't require a GPU. `src/data_loader.py` also implemented (`load_edit_chains()`, `load_image()`) with 2 new unit tests (12/12 total passing). **Not yet run** — no GPU on this machine, so this code is unverified against the real models until it runs on Colab.
- **Day 10-13 — Colab notebook**: `notebooks/colab_run_pipeline.ipynb` rewritten into a real, runnable pipeline. Now GitHub-based: `data/raw_images/` (60 images, 8.6MB) is committed to git (`.gitignore` updated to stop excluding it) so the notebook does a plain `git clone` of the repo instead of a manual zip upload. Repo is being made public on GitHub so Colab doesn't need a token. Notebook then loads the dataset, runs a SAM smoke test, and runs all 120 baseline edit chains (resizing to 512x512, saving every step to `results/baseline/`, skip-if-done so it survives a disconnect).

## Next
- **You**: make the GitHub repo public, open `notebooks/colab_run_pipeline.ipynb` in Google Colab (T4 GPU runtime), run it top to bottom. This is a real run against real models — expect it to take a while (120 chains x ~4 steps) and to need debugging on first run since none of this GPU code has executed yet.
- **Day 14-15** (after the Colab run): implement the drift-scoring pass over the saved baseline outputs using `drift_score.py` + `segment.py`, and spot-check scores by eye before trusting the full batch.

## Decisions / notes
- Implementation code lives in `Implementation/` (sibling to `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/`, `Final Paper/`), not at the repo root.
- Downloaded source PDFs live in `Research Papers - Existing/` and are gitignored (root `.gitignore`).
- Following the proposal's own Section 8/13.3 plans, but working through the 30-day timeline in order rather than jumping straight to code.
- **Design finalized (Day 1-3):**
  - Dataset source: ~60 images from COCO val2017, pulled via a reproducible script (not manual selection) — clear research-use licensing, matches Section 11 ethics notes.
  - Chain instructions: per-image custom instructions (not a fixed template pool applied to every image) — more realistic edits, but means instructions can only be written after images are selected, so this is now coupled to Session 3/4 rather than decided in the abstract.

## Blockers
- None.
