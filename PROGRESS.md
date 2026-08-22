# Progress

Overall status for the semantic-drift research project. Following the 30-day timeline in
[`Research Proposal/research_proposal_semantic_drift.md`](Research%20Proposal/research_proposal_semantic_drift.md) (Section 8), worked in order rather than jumping straight to code.
For code-session-level detail, see [`Implementation/PROGRESS.md`](Implementation/PROGRESS.md).

## Status: Days 1-15 complete, verified, and spot-checked; Day 16-19 (mitigations) written, ready to run on Colab

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

- **Day 10-13 — GPU code written and run**: `Implementation/src/edit_runner.py` (InstructPix2Pix), `src/segment.py` (SAM), `src/data_loader.py` implemented and run successfully on Colab GPU. All 120 baseline chains (60 images × Chain A + Chain B, 4 steps each) generated and downloaded — verified complete locally, now committed to git along with the source images.
- **Day 14-15 — Drift scoring**: `src/clip_embed.py` added and a real bug fixed (transformers 5.x changed CLIP's output format). First real Colab run crashed at chain 16/120 on a sparse-region edge case and lost all progress since the script only saved at the very end — fixed to write incrementally and skip already-scored chains on resume, verified locally with mocked failure injection. Second Colab run completed cleanly: 120/120 chains scored. Spot-check confirmed the metric works — lowest-drift chain genuinely only changed what was asked, highest-drift chain is a real catastrophic InstructPix2Pix failure (collapses to near-black by step 3) that's a strong illustrative example for the write-up. See `Implementation/PROGRESS.md` for full detail, including an early (non-final) observation that per-step drift looks roughly flat across chain position rather than increasing — worth the real statistical test before drawing conclusions.
- **Day 16-19 — Mitigations**: `src/mitigation.py` implemented (`region_locking`: post-hoc pixel revert outside the target box; `masked_conditioning`: crops to just the target region before generation so the model never sees the rest of the image). Both verified locally with mocked models (no GPU needed for the compositing logic itself). Shared scoring logic extracted into `src/drift_pipeline.py` so it isn't duplicated 3 ways (baseline + 2 strategies). Colab notebook updated with generation + scoring cells for both strategies. **Not yet run** — needs Colab GPU.

## Next
- **You**: run the new mitigation cells in `notebooks/colab_run_pipeline.ipynb` on Colab, download all three score CSVs (baseline, region_locking, masked_conditioning).
- **Day 22-23** (after that): `stats.py` — paired significance tests comparing baseline vs. each mitigation.

## Blockers
- None.
