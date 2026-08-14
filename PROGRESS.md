# Progress

Overall status for the semantic-drift research project. Following the 30-day timeline in
[`Research Proposal/research_proposal_semantic_drift.md`](Research%20Proposal/research_proposal_semantic_drift.md) (Section 8), worked in order rather than jumping straight to code.
For code-session-level detail, see [`Implementation/PROGRESS.md`](Implementation/PROGRESS.md).

## Status: Days 1-9 complete (dataset fully built)

## Done
- **Repo structure**: `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/` (gitignored PDFs), `Implementation/`, `Final Paper/`.
- **Day 1-3 — Scaffold repo**: `Implementation/` fully scaffolded (stub modules, config, requirements, tests dir, notebooks dir, `CLAUDE.md`). See `Implementation/PROGRESS.md`.
- **Day 1-3 — Literature review**: [`Literature Review/literature_review.md`](Literature%20Review/literature_review.md) written from the 12 papers in `Research Papers - Existing/`. Sharpened the gap statement: MagicBrush (NeurIPS 2023) already has multi-turn edit data but doesn't isolate non-target-region change; MasaCtrl and Plug-and-Play (both 2023) are the closest prior art to the masked-conditioning/region-locking mitigations but are generation-time interventions, not measurement tools.
- **Day 1-3 — Finalize design**: two open decisions locked (recorded in `Implementation/PROGRESS.md`):
  - Dataset source: ~60 images from COCO val2017, via a reproducible download script.
  - Chain instructions: per-image custom instructions (not a fixed template pool) — couples instruction-writing to image selection.

- **Day 4-5 — Drift Score logic**: `Implementation/src/drift_score.py` implemented and unit-tested (10/10 passing) using dummy vectors and a fake embedding function — no real CLIP/SAM calls yet. See `Implementation/PROGRESS.md` for details.

- **Day 6-9 — Dataset download**: 60 COCO val2017 images downloaded into `Implementation/data/raw_images/` via `scripts/download_coco_subset.py`, with a manifest (`data/coco_subset_manifest.json`) of each image's object categories.
- **Day 6-9 — Edit instructions**: every image individually viewed and given a custom Chain A (object-level, 4 steps) and Chain B (global, 4 steps) in `data/edit_instructions.json` — 120 chains total, verified complete against the manifest. Dataset construction is done.

## Next
- **Day 10-13**: `edit_runner.py` (InstructPix2Pix) + `segment.py` (SAM) on Colab GPU, then run baseline edit chains over the full dataset.

## Blockers
- None.
