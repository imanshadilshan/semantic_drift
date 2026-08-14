# Progress

## Status: Day 1-3 (repo scaffold + literature review) mostly done; design finalization in progress

## Done
- Repo scaffolding: folder structure under `Implementation/` (src, tests, data, configs, notebooks, results, human_eval).
- `requirements.txt`, `configs/default.yaml`, `data/edit_instructions.json` (schema example) in place.
- All `src/` modules stubbed with docstrings and `raise NotImplementedError` bodies.
- `src/utils.py` has working `get_device()` / `set_seed()`.
- `.gitignore` set up to keep `raw_images/`, `results/`, and model checkpoints out of git.
- Literature review written: `../Literature Review/literature_review.md`, based on the 12 downloaded PDFs in `../Research Papers - Existing/`. Key finding: MagicBrush (NeurIPS 2023) already has multi-turn edit data — good validation of the Chain A/B design, but its evaluation doesn't isolate collateral/non-target-region change, which is still the gap this project fills. MasaCtrl and Plug-and-Play are the closest prior art to the masked-conditioning/region-locking mitigations.

## Next
- Session 2 (Days 4-5): implement `src/drift_score.py` (`compute_drift_score()`, `compute_chain_drift_score()`) and `tests/test_drift_score.py` with dummy masks/fake embeddings, no real model calls yet — doesn't depend on the dataset being ready.
- Session 3/4 (Days 6-9): write the COCO val2017 download script, select ~60 images, then write per-image custom instruction chains (Chain A/B) once images are in hand — needs to see each image's content first.

## Decisions / notes
- Implementation code lives in `Implementation/` (sibling to `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/`, `Final Paper/`), not at the repo root.
- Downloaded source PDFs live in `Research Papers - Existing/` and are gitignored (root `.gitignore`).
- Following the proposal's own Section 8/13.3 plans, but working through the 30-day timeline in order rather than jumping straight to code.
- **Design finalized (Day 1-3):**
  - Dataset source: ~60 images from COCO val2017, pulled via a reproducible script (not manual selection) — clear research-use licensing, matches Section 11 ethics notes.
  - Chain instructions: per-image custom instructions (not a fixed template pool applied to every image) — more realistic edits, but means instructions can only be written after images are selected, so this is now coupled to Session 3/4 rather than decided in the abstract.

## Blockers
- None.
