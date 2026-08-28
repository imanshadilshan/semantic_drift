# Detecting and Mitigating Semantic Drift in Multi-Turn Instruction-Based Image Editing

## Abstract

Instruction-based image editors such as InstructPix2Pix let users apply a sequence of natural-language edits to an image, but nothing in the standard evaluation toolkit measures what happens to the parts of the image nobody asked to change. This project introduces the **Drift Score**, a model-agnostic metric combining Segment Anything (SAM) region segmentation with CLIP embedding similarity to quantify unintended change after an edit, and uses it to run an empirical study across 120 four-step edit chains (60 COCO images, split evenly between localized object-level edits and broad scene-level edits). Two low-cost mitigation strategies — masked conditioning and region-locking — were implemented and evaluated against an unmitigated baseline. Both mitigations produce large, highly significant reductions in measured drift (region-locking: 89.1%, p < 0.0001; masked conditioning: 68.6%, p < 0.0001), directly supporting RQ3. Two pre-registered hypotheses did not hold: region-locking significantly *outperformed* masked conditioning rather than the reverse (H2), and drift did not increase across the chain — the only significant step-position effect found ran in the opposite direction of the predicted compounding pattern (H1). Both reversals are explained mechanistically rather than left as unexplained anomalies. The Drift Score, dataset, and full pipeline are released as a reproducible toolkit.

## 1. Introduction

Iterative refinement — "make it sunset," then "add a bird," then "remove the fence" — is a natural usage pattern for instruction-based image editors, but it introduces a failure mode that single-step evaluation cannot see: a model can follow every individual instruction correctly while still drifting further and further from the user's actual intent, because nothing constrains it to leave the rest of the image alone. Existing evaluation protocols for these models (CLIP-based instruction-following scores, human preference studies) measure whether the requested change happened, not what else changed as a side effect. This project terms that side effect **semantic drift** and asks three questions: can it be measured automatically (RQ1), does it compound across a chain of edits (RQ2), and can cheap, pretrained-model-only mitigations reduce it (RQ3)?

This work uses only pretrained models: a laptop for development and analysis, and Google Colab for the heavier inference stages. No model was trained or fine-tuned; the contribution is the measurement methodology and the empirical findings it produces, not a new editing model.

## 2. Related Work

This project builds on five established lines of work, detailed further in the standalone [literature review](../Literature%20Review/literature_review.md):

- **InstructPix2Pix** (Brooks, Holynski & Efros, CVPR 2023) is the editor evaluated throughout this study. It is trained entirely on synthetically generated (image, instruction, edited-image) triplets, which means it inherits no explicit supervision for "leave everything else untouched" beyond whatever its training data's generation process (Prompt-to-Prompt attention injection, applied to *synthetic* image pairs) happened to preserve. This is the mechanistic reason drift is expected, not merely an empirical curiosity.
- **Latent Diffusion Models** (Rombach et al., CVPR 2022) are the generative backbone underlying InstructPix2Pix and nearly every other model in this review.
- **CLIP** (Radford et al., ICML 2021) supplies the embedding space the Drift Score measures change in, and the text-image similarity signal used to identify which region an instruction targets.
- **Segment Anything** (Kirillov et al., ICCV 2023) supplies the region decomposition without which drift could only be measured at whole-image granularity, which would blur out exactly the localized side effects this project targets.
- **Prompt-to-Prompt** (Hertz et al., 2022), **Plug-and-Play** (Tumanyan et al., CVPR 2023), and **MasaCtrl** (Cao et al., ICCV 2023) are generation-time consistency techniques — the closest prior art to this project's mitigation strategies. Critically, all three are interventions that aim to *produce* a more consistent edit; none of them measure how much unintended change remains, and none address chains of edits. MasaCtrl's own motivation section states the semantic-drift problem almost exactly, three years before this project, but stops at fixing single edits.
- **MagicBrush** (Zhang et al., NeurIPS 2023) already contains multi-turn edit data, validating this project's 4-step chain design as a realistic usage pattern — but its evaluation protocol only checks whether each turn's *target* region changed correctly, never how much everything else changed.

**The gap**: no existing work combines segmentation and embedding comparison into a chain-aware measurement of unintended change, and no existing work empirically compares mitigation strategies on that specific axis. This project's contribution is that combination, plus the empirical study it enables.

## 3. Methodology

### 3.1 The Drift Score

For a single edit step, given the pre-edit image, the post-edit image, and the instruction:

1. Segment the **pre-edit** image with SAM (`vit_b` checkpoint, `points_per_side=16`, `min_mask_region_area=500`, kept modest to fit Colab memory/time budgets), producing a set of region bounding boxes.
2. Identify the **target region** — the one box the instruction most plausibly refers to — by embedding each region crop and the instruction text with CLIP (`clip-vit-base-patch32`) and taking the highest cosine similarity.
3. Crop **both** the pre- and post-edit image at the *same* box coordinates for every non-target region. (Segmenting the post-edit image independently, rather than reusing the pre-edit boxes, would give unrelated region IDs with no correspondence between them — this was a deliberate design decision, not an oversight.)
4. For each non-target region, compute `1 − cosine_similarity(CLIP(pre_crop), CLIP(post_crop))`. The mean over all non-target regions is that step's Drift Score.
5. A chain's **cumulative Drift Score** is the sum of its per-step scores.

The full implementation is in `Implementation/src/drift_score.py` (pure logic, unit-tested with dummy vectors — no model calls) and `Implementation/src/segment.py` / `src/clip_embed.py` (the SAM/CLIP wrappers).

### 3.2 Dataset

60 images were sampled from COCO val2017 (`Implementation/scripts/download_coco_subset.py`, seed 42, reproducible), filtered to those with 2-8 annotated objects — enough to have both a plausible target and non-target regions without an overly cluttered scene. Each image received two hand-written, per-image (not templated) 4-instruction chains after being individually inspected:

- **Chain A (object-level)**: localized, single-object edits (e.g., "change the horse's color to black," "remove the striped poles in front of the horse").
- **Chain B (global)**: broad, scene-level edits (e.g., "make it a rainy day," "add golden sunset lighting").

This produced 120 chains total (60 object-level + 60 global), stored in `Implementation/data/edit_instructions.json`.

### 3.3 Pipeline and Models

| Component | Model | Role |
|---|---|---|
| Editor | `timbrooks/instruct-pix2pix` | Executes each instruction, 512×512 |
| Segmentation | SAM `vit_b` (~375MB) | Region decomposition for scoring and mitigation |
| Embedding | CLIP `ViT-B/32` | Region change measurement + target identification |

All three are pretrained, publicly available checkpoints, downloaded automatically on first use — no training occurred. Heavy inference (editing, segmentation) ran on Google Colab's GPU; drift-score aggregation and statistical analysis ran locally on CPU. Every image, edit chain, and score is committed to the project's git history for reproducibility.

### 3.4 Mitigation Strategies

Two strategies were implemented in `Implementation/src/mitigation.py`, both identifying the target region via the identical SAM+CLIP procedure the Drift Score itself uses:

- **Region-locking**: runs the edit on the full image as normal, then reverts every pixel outside the target region's box back to the pre-edit image — a post-hoc correction of whatever the model produced.
- **Masked conditioning**: crops down to just the target region (plus a 32-pixel padding margin for context) *before* generation, runs the edit only on that crop, then pastes the result back — a generation-time constraint, since the model never sees the rest of the image at all.

A third, stretch-goal strategy (attention-restricted editing, informed by Prompt-to-Prompt-style cross-attention control) was scoped in the original proposal as optional and was not implemented — RQ3 already has a clear, statistically-supported answer without it.

### 3.5 Statistical Analysis

Paired t-tests and Wilcoxon signed-rank tests (`Implementation/src/stats.py`, wrapping `scipy.stats`) compare cumulative Drift Scores between conditions, pairing by `(image_id, chain_type)`. Wilcoxon is treated as the primary test given the small, likely-skewed sample; the t-test is reported alongside it for completeness. 13 of 480 baseline steps (2.7%) could not be scored — SAM occasionally finds too few regions in a heavily stylized pre-edit image for anything to remain once the target region is excluded — and were dropped from pairwise comparisons rather than imputed.

## 4. Results

### 4.1 RQ1 — Can drift be measured automatically?

Qualitative spot-checks across the score range confirm the metric tracks real, visible content preservation. The lowest-scoring baseline chain (0.082, a surfer riding a wave) shows only the requested outfit and surfboard color changes — the wave, spray, and sky are pixel-for-pixel the same scene. The highest-scoring baseline chain (0.565, a boy holding a baseball glove) is a genuine catastrophic failure: by step 3 the image has collapsed to near-total black after step 2's edit destroyed the original composition, and step 4 edits on top of the wreckage. The score correctly flags this as the single worst chain in the dataset.

### 4.2 RQ3 — Do the mitigations reduce drift?

**Yes, substantially, in every breakdown tested** (Table 1). Both mitigations reduce mean cumulative drift by a wide margin relative to the unmitigated baseline (mean 0.293, range 0.082–0.565, n=120), and the effect holds separately for object-level and global chains.

**Table 1. Cumulative Drift Score by condition**

| Condition | Mean | Reduction vs. baseline | Wilcoxon p |
|---|---|---|---|
| Baseline | 0.293 | — | — |
| Masked conditioning | 0.092 | 68.6% | < 0.0001 |
| Region-locking | 0.032 | 89.1% | < 0.0001 |

Region-locking also significantly outperforms masked conditioning head-to-head (p < 0.0001). Broken down by chain type, both mitigations work slightly better on global chains than object-level chains (region-locking: 92.2% vs. 86.0% reduction), though both remain highly significant in both subgroups.

**On the same worst-case chain from 4.1**: baseline cumulative drift 0.565 falls to 0.076 under region-locking and 0.180 under masked conditioning — both large recoveries, visually confirmed (Section 5.2).

### 4.3 RQ2 — Does drift compound across the chain?

No consecutive step transition is statistically significant (step 1→2: p=0.85; 2→3: p=0.33; 3→4: p=0.47) — there is no detectable escalation at any single point in the chain. Comparing the two endpoints directly, step 1 vs. step 4 *is* significant (p=0.012 t-test, p=0.006 Wilcoxon), but with step 4 showing **less** drift (mean 0.066) than step 1 (mean 0.080) — the reverse of the predicted direction.

## 5. Discussion

### 5.1 H1 did not hold, and the explanation matters

The working hypothesis was that later edits would show progressively more collateral damage because each operates on an already-modified image. The data shows the opposite at the only point where a significant difference exists at all. The most plausible explanation, consistent with the baseball-glove example in 4.1: a chain that collapses catastrophically early (as several evidently do) has, by construction of a CLIP-similarity metric, very little room left to register as "further changed" in subsequent steps — an image that is already almost entirely black cannot become much more different from its original self than it already is. This is a **measurement ceiling effect masking front-loaded failure**, not evidence that later edits are genuinely gentler. Distinguishing "drift is front-loaded and severe" from "drift doesn't compound" would require a metric sensitive to *further* degradation of an already-degraded image — a natural direction for follow-up work, not something this project's CLIP-similarity approach can resolve as built.

### 5.2 H2 reversed, and the reversal has a mechanistic explanation

The prediction was that masked conditioning — a stricter, generation-time constraint — would outperform region-locking's after-the-fact correction, at some cost to edit quality at the boundary. Region-locking instead significantly outperforms masked conditioning on the drift metric itself. The most likely reason is a property of the metric-mitigation interaction rather than either mitigation being fundamentally weaker: masked conditioning edits a *padded* crop (target box plus a 32px margin, deliberately included so the model has enough surrounding context to produce a coherent edit), but the Drift Score only excludes the raw target box from scoring — so masked conditioning's own padding ring is counted as drift even when nothing objectionable happened there. Region-locking's hard revert has no equivalent margin. This should not be read as proof that region-locking produces better-*looking* edits in general — the visual spot-check in Section 4 found region-locking has its own artifact (a visible, unblended rectangular seam near the glove in the worst-case example), consistent with the boundary-cost intuition behind the original H2, just not showing up as a *drift-score* cost the way the hypothesis predicted.

### 5.3 A shared weak point: target-region identification

Both mitigations, and the Drift Score itself, depend on CLIP correctly identifying which region an instruction targets. This works reliably for "change X" and "remove X" instructions, where the target already exists in the pre-edit image. It does not work reliably for "add X" instructions, where nothing pre-existing can match well. Tracing one concrete failure directly: for the instruction "add a cap on the boy's head" (step 4 of the worst-case chain), the identified target box covered 99.6% of the frame — meaning masked conditioning's spatial constraint provided no real protection for that step, since it was cropping to almost the entire image anyway. Region-locking faced the identical oversized box for the same step and happened to produce a fine result, most likely because diffusion sampling is stochastic (no fixed seed) — the two mitigations' generation calls on that step were independent random draws from the same model, and one landed conservative while the other did not. This is a real, reproducible limitation of the target-identification approach, not a bug in either mitigation's implementation, and it affects roughly half of the object-level chains' instructions (those with "add" verbs).

## 6. Limitations

- **Dataset size** (60 images, 120 chains) is adequate for the paired statistical tests reported here but should be read as an initial empirical study, not a definitive benchmark, per the original proposal's framing.
- **Single editor model.** All results are specific to InstructPix2Pix. The Drift Score is designed to be model-agnostic, but that design goal was not empirically tested against a second editor within this project's scope — a natural, low-cost extension (see Section 7).
- **CLIP similarity is an imperfect proxy for human-perceived change**, as flagged in the original proposal. The planned human-perceptual sanity check (Section 5.5 of the proposal; the rating tool is built and published) was not carried out — this was an explicitly optional step and was skipped by choice under time constraints, but its absence means the CLIP-based drift numbers in this report carry that caveat without an independent check against human judgment.
- **Target-region identification via CLIP top-1 similarity is a known weak point for "add" instructions** (Section 5.3), affecting both the metric's accuracy and both mitigations' effectiveness for that instruction type specifically.
- **Region boxes, not exact segmentation masks**, are used throughout for both scoring and mitigation, for consistency and simplicity. This is a coarser spatial unit than SAM's actual per-pixel masks and is the direct cause of masked conditioning's padding-margin effect discussed in 5.2.
- **The attention-restricted editing mitigation (stretch goal) was not implemented**, per the proposal's own scoping — RQ3 has a clear answer without it.

## 7. Ethical Considerations

All base images are drawn from COCO val2017, licensed for research use. No images were selected for or found to depict identifiable individuals in a way that editing could plausibly harm; several chains do edit photographs containing people (e.g., changing clothing color, adding accessories), and no instruction was written to generate misleading or harmful depictions. The (unused) human-evaluation tool was designed to collect only numeric ratings and an optional rater name, with no other personal data.

## 8. Conclusion and Future Work

This project set out to answer whether unintended drift across a chain of instruction-based image edits could be measured automatically, whether it compounds, and whether cheap mitigations reduce it. RQ1 and RQ3 have clear, positive, and statistically well-supported answers: yes, and yes, by a wide margin. RQ2's answer is more interesting than a simple yes or no — the data suggests drift may be front-loaded into occasional catastrophic failures rather than gradually compounding, which the Drift Score as built cannot fully distinguish from "no compounding at all." Two specific, well-diagnosed mechanisms — a measurement ceiling effect and a metric/mitigation margin mismatch — explain why two of the original hypotheses reversed rather than simply failing to replicate, which is itself evidence the underlying measurement is capturing something real and interpretable rather than noise.

Natural next steps: evaluate the Drift Score against a second, more recent editing model to test its claimed model-agnosticism; run the built human-perceptual check to validate CLIP-based drift against human judgment; and investigate a drift metric less prone to the ceiling effect identified in 5.1, to separate "drift compounds gradually" from "drift arrives suddenly and then plateaus."

## References

Brooks, T., Holynski, A., & Efros, A. A. (2023). InstructPix2Pix: Learning to Follow Image Editing Instructions. *CVPR 2023*.

Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., & Zheng, Y. (2023). MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. *ICCV 2023*.

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D. (2022). Prompt-to-Prompt Image Editing with Cross Attention Control. *arXiv:2208.01626*.

Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. *ICCV 2023*.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.

Tumanyan, N., Geyer, M., Bagon, S., & Dekel, T. (2023). Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. *CVPR 2023*.

Zhang, K., Mo, L., Chen, W., Sun, H., & Su, Y. (2023). MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. *NeurIPS 2023, Datasets and Benchmarks Track*.
