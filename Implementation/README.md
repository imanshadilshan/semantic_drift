# Semantic Drift

Implementation for "Detecting and Mitigating Semantic Drift in Multi-Turn Instruction-Based Image Editing".
Full proposal: [`../Research Proposal/research_proposal_semantic_drift.md`](../Research%20Proposal/research_proposal_semantic_drift.md).

## Setup
```
pip install -r requirements.txt
```

## Structure
- `src/drift_score.py` — core Drift Score logic (CPU-only, unit-testable)
- `src/data_loader.py` — reads `data/edit_instructions.json` + `data/raw_images/`
- `src/edit_runner.py` — InstructPix2Pix wrapper (GPU)
- `src/segment.py` — SAM wrapper (GPU-preferred)
- `src/mitigation.py` — masked conditioning / region-locking / attention-restricted editing
- `src/stats.py` — paired significance tests
- `tests/` — unit tests, dummy data only
- `notebooks/colab_run_pipeline.ipynb` — GPU pipeline entry point for Colab
- `results/baseline/`, `results/mitigated/` — saved edit-chain outputs
- `human_eval/` — optional peer perceptual-rating results

See `PROGRESS.md` for current status and `CLAUDE.md` for conventions.
