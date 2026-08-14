# Progress

Overall status for the semantic-drift research project. Following the 30-day timeline in
[`Research Proposal/research_proposal_semantic_drift.md`](Research%20Proposal/research_proposal_semantic_drift.md) (Section 8), worked in order rather than jumping straight to code.
For code-session-level detail, see [`Implementation/PROGRESS.md`](Implementation/PROGRESS.md).

## Status: Days 1-3 complete

## Done
- **Repo structure**: `Research Proposal/`, `Literature Review/`, `Research Papers - Existing/` (gitignored PDFs), `Implementation/`, `Final Paper/`.
- **Day 1-3 — Scaffold repo**: `Implementation/` fully scaffolded (stub modules, config, requirements, tests dir, notebooks dir, `CLAUDE.md`). See `Implementation/PROGRESS.md`.
- **Day 1-3 — Literature review**: [`Literature Review/literature_review.md`](Literature%20Review/literature_review.md) written from the 12 papers in `Research Papers - Existing/`. Sharpened the gap statement: MagicBrush (NeurIPS 2023) already has multi-turn edit data but doesn't isolate non-target-region change; MasaCtrl and Plug-and-Play (both 2023) are the closest prior art to the masked-conditioning/region-locking mitigations but are generation-time interventions, not measurement tools.
- **Day 1-3 — Finalize design**: two open decisions locked (recorded in `Implementation/PROGRESS.md`):
  - Dataset source: ~60 images from COCO val2017, via a reproducible download script.
  - Chain instructions: per-image custom instructions (not a fixed template pool) — couples instruction-writing to image selection.

## Next
- **Day 4-5**: implement `Implementation/src/drift_score.py` (`compute_drift_score()`, `compute_chain_drift_score()`) + `tests/test_drift_score.py` using dummy masks/fake embeddings — no dataset dependency.
- **Day 6-9**: write the COCO val2017 download script, select ~60 images, then write per-image Chain A/B instructions.

## Blockers
- None.
