# Progress

## Status: Days 1-13 code done and run (edit chains generated on Colab GPU); Day 14-15 drift-scoring code fixed for resilience, re-running on Colab now

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

- **Day 10-13 — Colab run complete**: repo made public, notebook run on Colab GPU, all 120 baseline chains generated (5 images each: original + 4 steps) and downloaded back — verified locally, all 120 chain folders complete, no gaps. `results/baseline/` is now committed to git too (`.gitignore` updated) alongside `data/raw_images/`.
- **Day 14-15 — Drift scoring**: `src/clip_embed.py` added (`embed_image()`, `embed_text()`, `identify_target_regions()` — CLIP picks which pre-edit region an instruction targets, via cosine similarity to the instruction text). `src/segment.py` refactored: `get_region_boxes()` + `crop_regions()` split out so the SAME box set (from the pre-edit image) is used to crop both pre- and post-edit images — segmenting each independently would give unrelated region_ids with no correspondence. `scripts/compute_baseline_drift.py` written, ties it together into `results/baseline_drift_scores.csv`.
  - **Bug found and fixed during smoke test**: `transformers` 5.x changed `CLIPModel.get_image_features()`/`get_text_features()` to return a `BaseModelOutputWithPooling` instead of a plain tensor — the real 512-dim embedding is in `.pooler_output`, not the return value itself. Verified the fix with a synthetic red/blue image vs. text test (correct image matched correct text: 0.31 vs 0.24 similarity).
  - **Known limitation, not a bug**: `identify_target_regions()` matches instruction text against *pre-edit* regions, so it works well for "remove X" / "change X" instructions but has no strong match for "add X" instructions (the new object doesn't exist pre-edit) — meaning "add" steps will show somewhat inflated drift, since the newly-added (legitimately requested) content isn't excluded as a target region. Worth a sentence in the write-up's Limitations section.
  - Single-chain smoke test passed with plausible scores (~0.06/step). **CPU speed is the blocker**: ~178s/chain measured locally → ~6 hours for all 120, so this needs to run on Colab GPU, not the laptop.
  - **Real run on Colab (first attempt) crashed at chain 16/120** with `ValueError: No non-target regions available to score drift over` — for one image (teddy bears, global chain) after a heavy stylization step, SAM found too few regions in the pre-image for anything to be left over once the target region was excluded. First 15 chains scored fine (plausible values, e.g. cumulative 0.10-0.49 range) before the crash.
  - **Two resilience bugs fixed as a result**: (1) `main()` only wrote the CSV once at the very end, so the crash lost all 15 already-scored chains' work — now writes incrementally, one chain at a time, via `append_rows()`. (2) A single unscoreable step now gets caught, recorded as `None` in the CSV, and the chain continues (rather than the whole 120-chain run dying on one problem image) — `compute_chain_drift_score()` still runs on whichever steps *did* succeed. Also added skip-already-scored resume logic (`load_already_scored()`), so re-running after a crash doesn't repeat already-completed chains. Both fixes verified locally with mocked failure injection before pushing (full run is too slow on CPU to re-verify end-to-end locally).

## Next
- **You**: re-run the "Compute Drift Scores" cell in Colab (git pull picks up the fix). The old script never wrote partial output, so this restarts from chain 1 — but now it survives sparse-region images instead of crashing, and if it does stop for any other reason, re-running will skip whatever chains already made it into the CSV. Download `results/baseline_drift_scores.csv` when done.
- **Day 16-19** (after that): implement and run the masked-conditioning + region-locking mitigations, using the baseline scores as the comparison point.

## Decisions / notes
- Implementation code lives in `Implementation/` (sibling to `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/`, `Final Paper/`), not at the repo root.
- Downloaded source PDFs live in `Research Papers - Existing/` and are gitignored (root `.gitignore`).
- Following the proposal's own Section 8/13.3 plans, but working through the 30-day timeline in order rather than jumping straight to code.
- **Design finalized (Day 1-3):**
  - Dataset source: ~60 images from COCO val2017, pulled via a reproducible script (not manual selection) — clear research-use licensing, matches Section 11 ethics notes.
  - Chain instructions: per-image custom instructions (not a fixed template pool applied to every image) — more realistic edits, but means instructions can only be written after images are selected, so this is now coupled to Session 3/4 rather than decided in the abstract.

## Blockers
- None.
