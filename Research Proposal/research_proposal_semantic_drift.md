# Research Proposal

## Title
**Detecting and Mitigating Semantic Drift in Multi-Turn Instruction-Based Image Editing**

## Duration
30 days — compute: laptop (development/analysis) + free-tier Colab GPU (heavy inference)

---

## Abstract

Instruction-based image editing models let users apply a sequence of natural-language edits to an image — "make the sky sunset," then "add a bird," then "remove the fence." Ideally, each instruction only changes the region it refers to. In practice, these models frequently introduce unintended changes elsewhere in the image, an effect this project terms **semantic drift**. No lightweight, automated method currently exists to measure how much unintended drift accumulates across a chain of edits, or to compare editing strategies on this axis. This project proposes (1) a reusable, model-agnostic **Drift Score**, built by combining CLIP-based semantic similarity with SAM-based region segmentation, (2) an empirical study of how drift compounds across multi-step edit chains, and (3) an evaluation of low-cost mitigation strategies (masked conditioning, region-locking) for reducing drift. The project uses only pretrained, freely available models — no training from scratch — making it feasible within a 30-day, laptop-plus-free-Colab-GPU budget.

## 1. Introduction & Motivation

Diffusion-based image editors (e.g., InstructPix2Pix, and SDXL-based editing pipelines) have made natural-language image editing broadly accessible. A common real-world usage pattern is iterative: users refine an image through several small edits rather than one large prompt. This makes the *consistency* of editing — how well the model preserves everything not mentioned in the instruction — just as important as the *accuracy* of the requested change.

Most existing evaluation protocols for these models (e.g., CLIP-based instruction-following scores) measure whether the requested change happened, but not what else changed as a side effect. This gap matters because unintended drift compounds: a small unwanted shift after edit 1 can be amplified by edit 2, and so on, eventually producing an image far removed from the user's intent even though every individual instruction was followed "successfully" by conventional metrics.

This project is scoped for a single contributor, pretrained-model-only pipeline, and free-tier compute, in order to be realistically completed in 30 days while still producing a result worth writing up and potentially extending into a longer thesis.

## 2. Related Work (Background)

This project builds directly on three established lines of work rather than developing new architectures:

- **Instruction-based image editing**: Brooks, Holynski, and Efros introduced InstructPix2Pix, which fine-tunes a diffusion model to follow natural-language edit instructions directly, without requiring a full text description of the target image (Brooks et al., CVPR 2023).
- **Latent diffusion models**: The underlying generative backbone for most modern instruction-editors is the latent diffusion architecture introduced by Rombach et al. (Stable Diffusion, CVPR 2022).
- **Attention-based edit localization**: Hertz et al. showed that cross-attention maps inside diffusion models can be manipulated to control which regions of an image are affected by a prompt edit (Prompt-to-Prompt, 2022) — directly relevant to the masked-conditioning mitigation strategy proposed here.
- **Vision-language similarity**: Radford et al.'s CLIP model provides a general-purpose way to measure semantic similarity between images and text, or between two images via their embeddings (Radford et al., ICML 2021).
- **Promptable segmentation**: Kirillov et al.'s Segment Anything (SAM) provides zero-shot, promptable image segmentation, used here to divide each image into regions so drift can be measured locally rather than globally (Kirillov et al., ICCV 2023).

**Gap this project addresses**: none of the above directly measures *unintended* regional change across a *chain* of edits. Existing instruction-following metrics are single-step and target-region-only. This project's contribution is the combination of segmentation + embedding comparison into a chain-aware drift metric, plus an empirical test of mitigation strategies.

## 3. Problem Statement

There is no lightweight, automated way to measure how much unintended drift accumulates across a chain of instruction-based image edits, and no empirical comparison of low-cost mitigation strategies for reducing it.

## 4. Research Questions & Hypotheses

1. **RQ1**: Can an automatic, reproducible score capture unintended regional change after a single edit or a chain of edits?
2. **RQ2**: How does drift accumulate across sequential edits — linearly, or does it compound faster with chain length?
3. **RQ3**: Do simple, low-cost mitigation strategies (masked conditioning, region-locking) meaningfully reduce drift without degrading the requested edit's success?

**Working hypotheses**:
- H1: Drift will increase with chain length, and the increase will be super-linear (later edits will show more collateral change than earlier ones) because each edit operates on an already-modified image.
- H2: Masked conditioning will reduce drift more effectively than region-locking, but at some cost to edit quality at region boundaries (visible seams).

## 5. Proposed Methodology

### 5.1 Dataset Construction
- 50-80 base images, drawn from an existing open dataset (e.g., a COCO subset) to save collection time.
- For each image, define **two edit-chain types**:
  - *Chain A - Object-level edits*: e.g., "add a bird," "remove the fence," "change the dog's collar color" (localized, single-object edits).
  - *Chain B - Global edits*: e.g., "make it sunset," "add fog," "make it look like winter" (broad, scene-level edits).
- Each chain: 4-5 sequential instructions, applied in order, with every intermediate output saved.
- This split lets the study compare drift behavior between localized vs. global edit types - an analysis the original 20-day scope didn't have room for.

### 5.2 Edit Chain Generation
- Run every image through InstructPix2Pix (or a comparable open editor) for each instruction in its chain, saving outputs at every step.
- Run this both **with** and **without** mitigation applied (see 5.4), producing paired baseline/mitigated chains for direct comparison.

### 5.3 Drift Scoring Pipeline
For each edit step:
1. Segment the pre- and post-edit image into regions using SAM.
2. Identify the target region(s) via CLIP similarity between the instruction text and region crops.
3. For all non-target regions, compute an embedding-distance-based change score between pre- and post-edit crops.
4. Aggregate into a per-step **Drift Score**, and a cumulative **Chain Drift Score** across the full sequence.

### 5.4 Mitigation Strategies (expanded set, enabled by the extra 10 days)
1. **Masked conditioning** - restrict the diffusion edit to the target region using the SAM mask.
2. **Region-locking** - re-composite untouched regions from the prior step after each edit.
3. **Attention-restricted editing** (stretch goal) - apply cross-attention control (informed by Prompt-to-Prompt-style manipulation) to bias the model's attention toward the target region during generation, without hard-masking pixels.

### 5.5 Statistical & Qualitative Analysis
- Compare Drift Scores (baseline vs. each mitigation strategy) using paired statistical tests (e.g., paired t-test or Wilcoxon signed-rank, given the small sample) to check whether differences are meaningful and not just noise.
- Break down results by chain type (object-level vs. global) and by chain position (edit 1 vs. edit 4) to test H1 and H2.
- Optional lightweight human check: ask 3-5 peers to rate a subset of ~15 image pairs (baseline vs. mitigated) on perceived unintended change, as a sanity check against the automated score - feasible now with the extra time, and strengthens the write-up considerably.

## 6. Development Environment & Workflow (Hybrid Setup)

### Why hybrid
- **Laptop**: all logic development, unit testing, dataset prep, statistical analysis, and plotting - no GPU required.
- **Colab (free-tier GPU)**: reserved for the heavy inference - running InstructPix2Pix and SAM over the real dataset.

Device-agnostic code (`device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`) means the same scripts run (slowly) on the laptop for testing and (fast) on Colab for real runs, avoiding duplicated code paths.

### Project structure
```
semantic-drift/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml
├── data/
│   ├── raw_images/
│   └── edit_instructions.json      # object-level + global chains
├── src/
│   ├── data_loader.py
│   ├── edit_runner.py              # InstructPix2Pix calls - heavy, GPU-only
│   ├── segment.py                  # SAM calls - heavy, GPU-preferred
│   ├── drift_score.py              # pure logic - CPU-friendly, unit-testable
│   ├── mitigation.py               # masked conditioning / region-locking / attention control
│   ├── stats.py                    # paired significance tests
│   └── utils.py
├── notebooks/
│   └── colab_run_pipeline.ipynb
├── tests/
│   └── test_drift_score.py
├── human_eval/
│   └── rating_form_results.csv     # optional peer perceptual check
└── results/
    ├── baseline/
    └── mitigated/
```

### Core Drift Score (implementation-ready pseudocode)
```python
def compute_drift_score(pre_regions, post_regions, target_region_ids, embed_fn):
    """
    pre_regions / post_regions: dict {region_id: cropped_image}
    target_region_ids: region(s) the instruction was meant to affect
    embed_fn: function returning a CLIP embedding for an image crop
    """
    drift_scores = []
    for region_id in pre_regions:
        if region_id in target_region_ids:
            continue  # skip the region that was supposed to change
        pre_embed = embed_fn(pre_regions[region_id])
        post_embed = embed_fn(post_regions.get(region_id, pre_regions[region_id]))
        similarity = cosine_similarity(pre_embed, post_embed)
        drift_scores.append(1 - similarity)  # higher = more unintended change
    return average(drift_scores)
```

### Model & library specifics
- `diffusers` - InstructPix2Pix, checkpoint `timbrooks/instruct-pix2pix`
- `transformers` - CLIP, checkpoint `openai/clip-vit-base-patch32`
- `segment-anything` - checkpoint `vit_b` (~375MB), to keep memory manageable on free-tier Colab
- `torch`, `torchvision`
- `scipy` - for paired statistical tests
- `opencv-python`, `numpy`, `matplotlib`, `pandas`

```
torch
torchvision
diffusers
transformers
segment-anything
scipy
opencv-python
numpy
pandas
matplotlib
pyyaml
```

### Handing this off to Claude Code
Suggested first prompt:
> "Set up the semantic-drift project with this folder structure: [paste structure above]. Start with `drift_score.py` and its unit tests in `tests/test_drift_score.py`, using dummy segmentation masks and fake embeddings - no real model calls yet. Then draft `edit_runner.py` and `segment.py` as thin wrappers around InstructPix2Pix and SAM, `mitigation.py` for the three strategies, `stats.py` for paired significance testing, and a Colab notebook that ties it all together."

## 7. Evaluation Plan

| Metric | Purpose |
|---|---|
| Drift Score (per step, per chain) | Core novel metric - quantifies unintended change |
| Cumulative Chain Drift Score | Tests whether drift compounds across steps (RQ2) |
| Edit Success Score (CLIP-based) | Confirms requested edit still happened after mitigation |
| Paired significance test (baseline vs. mitigated) | Confirms observed differences aren't noise (RQ3) |
| Object-level vs. global chain comparison | Tests whether drift behavior differs by edit type |
| Optional human perceptual ratings | Sanity check against automated score |

## 8. 30-Day Timeline

| Days | Task | Where |
|---|---|---|
| 1-3 | Literature review write-up, finalize design, scaffold repo | Laptop |
| 4-5 | Implement `drift_score.py` + `stats.py` with unit tests on dummy data | Laptop |
| 6-9 | Build dataset: 50-80 images, object-level + global edit chains | Laptop |
| 10-13 | Integrate real model calls (InstructPix2Pix, SAM); run full baseline edit chains | Colab GPU |
| 14-15 | Collect and validate baseline Drift Scores | Colab GPU |
| 16-19 | Implement and run masked conditioning + region-locking mitigations | Colab GPU |
| 20-21 | (Stretch) Implement and run attention-restricted editing mitigation | Colab GPU |
| 22-23 | Statistical analysis: paired tests, chain-position and chain-type breakdowns | Laptop |
| 24-25 | Optional human perceptual check (3-5 raters, ~15 image pairs) | Laptop |
| 26-28 | Full write-up: results, figures, discussion | Laptop |
| 29 | Internal review pass - check claims against data, tighten writing | Laptop |
| 30 | Final polish, proofreading, prepare presentation/defense slides | Laptop |

## 9. Expected Contributions

1. A reusable, model-agnostic **Drift Score** for evaluating instruction-based image editors, open-sourced as a small toolkit.
2. Empirical evidence on how drift accumulates across multi-step edit chains, broken down by edit type (object-level vs. global).
3. A comparative evaluation of three low-cost mitigation strategies, with statistical backing rather than anecdotal comparison.
4. (If time allows) A small human-perception validation of the automated metric, strengthening confidence in its usefulness.

## 10. Limitations & Risks

- Dataset size (50-80 images) is still modest for strong statistical claims - results should be framed as an initial empirical study, not a definitive benchmark.
- Free-tier Colab GPU limits (memory, session timeouts) may still require batching or lower-resolution runs.
- CLIP-based similarity is an imperfect proxy for human-perceived "unintended change" - the optional human evaluation step exists specifically to check this.
- The attention-restricted editing mitigation (stretch goal) is more technically involved and may be dropped if days 20-21 run over budget elsewhere; the core contribution (RQ1-RQ3 with the first two mitigations) does not depend on it.
- Switching between laptop and Colab adds minor sync overhead - mitigated by keeping the repo Git-based from day one.

## 11. Ethical Considerations

- All base images should come from datasets licensed for research use (e.g., COCO) to avoid copyright issues.
- If any images depicting real people are included, edits and analysis should avoid generating or amplifying misleading or harmful depictions; synthetic or clearly non-identifying images are preferable where possible.
- If human raters are used for the optional perceptual check, keep the task low-burden and anonymous (no personal data collected beyond ratings).

## 12. Tools & Resources Needed

- Python, PyTorch (device-agnostic code, see Section 6)
- Pretrained: InstructPix2Pix, CLIP, Segment Anything (SAM) - all free via Hugging Face / official repos
- Google Colab (free tier) for GPU-dependent runs; personal laptop for development, testing, and analysis
- An open-licensed image dataset subset (e.g., COCO)
- Git/GitHub (or Google Drive) for syncing code and results
- 3-5 peers willing to spend ~15 minutes on an optional perceptual rating task

## 13. Full Implementation Plan (Claude Code Sessions + Dual-Account Strategy)

### 13.1 How Claude Pro usage limits work (and why two accounts help)
- Pro plan usage resets on a rolling 5-hour window that starts from your first message in a session, not a fixed clock time — and Claude Code draws from the same usage pool as regular Claude.ai chat, so a long agentic coding session can use up the window faster than expected.
- There is also a separate weekly cap on top of the 5-hour session limit.
- Practical implication: if Account A runs out mid-session, logging into Account B gives you a fully independent 5-hour window and weekly allowance — effectively doubling usable coding time on the most intensive days (the Colab-integration stretch, Days 10-21).
- Mechanics of switching: `claude logout` then `claude login` with the second account's credentials, or run Claude Code from two separate terminal profiles so both stay signed in and ready without re-authenticating each time.
- Check remaining headroom before starting a heavy session with `/status` inside Claude Code, or Settings > Usage on claude.ai — know which account has room before committing to a long task.
- Treat one account as primary and the second as overflow rather than splitting every session evenly in advance; this avoids both running low on the same day.

### 13.2 Keeping progress continuous across sessions and accounts
Two accounts means two separate Claude Code login contexts — nothing carries over automatically except what's committed to the repo. To avoid re-explaining context every switch:
- Keep a `PROGRESS.md` at the repo root, updated at the end of every session: what was finished, what's next, any decisions or blockers.
- Commit small, working changes frequently so a fresh session (on either account) can `git pull` and immediately see the true current state.
- Maintain a `CLAUDE.md` at the repo root — Claude Code reads this automatically at startup — summarizing project structure, conventions, and current status, so any new session has full context without manual re-briefing.

### 13.3 Session-by-session implementation checklist
Each session below is scoped to comfortably fit inside a single 5-hour window, so it can be assigned to whichever account has headroom that day.

| Session | Days | Where | Task | Sample prompt to Claude Code |
|---|---|---|---|---|
| 1 | 1-2 | Laptop | Repo scaffolding | "Initialize the semantic-drift repo with this structure: [paste]. Create requirements.txt, configs/default.yaml, stub files with docstrings for each module, plus PROGRESS.md and CLAUDE.md summarizing the project." |
| 2 | 3-4 | Laptop | Drift score logic + tests | "Implement drift_score.py per this pseudocode [paste]. Write tests/test_drift_score.py using dummy segmentation masks and fake embeddings — no real model calls. Run the tests and confirm they pass." |
| 3 | 5 | Laptop | Dataset schema + loader | "Implement data_loader.py to read data/edit_instructions.json (fields: image_id, chain_type, instructions[]) and raw_images/, with validation for missing files." |
| 4 | 6-9 | Laptop | Dataset construction | Curate/download 50-80 images and write Chain A / Chain B instructions manually; use Claude Code to validate and reformat the resulting JSON. |
| 5 | 10-11 | Colab (Account A) | Model wrappers | "Implement edit_runner.py wrapping the InstructPix2Pix diffusers pipeline (timbrooks/instruct-pix2pix) and segment.py wrapping SAM (checkpoint vit_b). Device-agnostic, callable from the Colab notebook." |
| 6 | 12-13 | Colab | Baseline runs | Run full baseline edit chains over the dataset; save intermediate outputs to results/baseline/. |
| 7 | 14-15 | Colab | Drift score collection | Run drift_score.py over baseline outputs; spot-check a handful of scores by eye before trusting the full batch. |
| 8 | 16-19 | Colab | Mitigation implementation + runs | "Implement mitigation.py with masked_conditioning() and region_locking() per Section 5.4. Re-run edit chains with each mitigation, saving to results/mitigated/." |
| 9 | 20-21 | Colab (stretch) | Attention-restricted editing | Only if on schedule; skip without risk to the core deliverable if time is tight. |
| 10 | 22-23 | Laptop | Statistics | "Implement stats.py with paired t-test and Wilcoxon signed-rank comparisons between baseline and each mitigation's Drift Scores, broken down by chain type and chain position." |
| 11 | 24-25 | Laptop | Optional human eval | Build a simple rating form for 3-5 peers; store results in human_eval/. |
| 12 | 26-28 | Laptop | Write-up | Draft the final report using results, figures, and stats output. |
| 13 | 29-30 | Laptop | Review & polish | Verify every claim in the write-up traces to a specific result file; proofread; prepare slides. |

### 13.4 Managing the highest-risk days
Days 10-21 carry the most usage risk, since first-pass model integration and debugging tend to be iterative. Practical approach:
- Start heavy integration days on whichever account has a full window available.
- If you hit a cap mid-debug, switch immediately rather than waiting — `git pull` on the other account and continue from the last commit.
- Watch the weekly cap as the tighter real constraint day-to-day; hitting the 5-hour cap twice in one day is unlikely for this schedule, but several consecutive heavy debugging days can approach the weekly limit on one account, which is exactly when the second becomes useful.

## 14. Selected References

- Brooks, T., Holynski, A., & Efros, A. A. (2023). InstructPix2Pix: Learning to Follow Image Editing Instructions. CVPR 2023.
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
- Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D. (2022). Prompt-to-Prompt Image Editing with Cross Attention Control. arXiv:2208.01626.
- Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
- Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. ICCV 2023.
