# CLAUDE.md

## Project
Implementation for the "Semantic Drift" research proposal (see `../Research Proposal/research_proposal_semantic_drift.md`).
Goal: a model-agnostic Drift Score (CLIP + SAM) measuring unintended regional change across chains of
instruction-based image edits, plus an evaluation of low-cost mitigation strategies.

## Status
Check `PROGRESS.md` first — it tracks what's done and what's next, session by session.

## Conventions
- Device-agnostic code: use `src.utils.get_device()`, never hardcode `"cuda"` or `"cpu"`.
- `src/drift_score.py` and `src/stats.py` are pure logic — CPU-only, no model loading, fully unit-testable.
- `src/edit_runner.py` and `src/segment.py` wrap heavy models (InstructPix2Pix, SAM) — GPU-only/preferred, run on Colab.
- Tests in `tests/` use dummy masks and fake embeddings — never call real models in unit tests.
- Config values (checkpoints, paths, chain settings) live in `configs/default.yaml`, not hardcoded in source.
- Large/generated artifacts (`data/raw_images/`, `results/`) are gitignored; only code and small JSON/YAML are committed.

## Workflow
- Update `PROGRESS.md` at the end of every session (done / next / blockers).
- Commit small, working changes frequently so any session — on either Claude account — can `git pull` and continue.
