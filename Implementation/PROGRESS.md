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

## Next
- **Day 10-13**: implement `src/edit_runner.py` (InstructPix2Pix wrapper) and `src/segment.py` (SAM wrapper) on Colab GPU, then run full baseline edit chains over the dataset.

## Decisions / notes
- Implementation code lives in `Implementation/` (sibling to `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/`, `Final Paper/`), not at the repo root.
- Downloaded source PDFs live in `Research Papers - Existing/` and are gitignored (root `.gitignore`).
- Following the proposal's own Section 8/13.3 plans, but working through the 30-day timeline in order rather than jumping straight to code.
- **Design finalized (Day 1-3):**
  - Dataset source: ~60 images from COCO val2017, pulled via a reproducible script (not manual selection) — clear research-use licensing, matches Section 11 ethics notes.
  - Chain instructions: per-image custom instructions (not a fixed template pool applied to every image) — more realistic edits, but means instructions can only be written after images are selected, so this is now coupled to Session 3/4 rather than decided in the abstract.

## Blockers
- None.
