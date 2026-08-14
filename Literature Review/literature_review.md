# Literature Review

## Detecting and Mitigating Semantic Drift in Multi-Turn Instruction-Based Image Editing

This review expands Section 2 of the [research proposal](../Research%20Proposal/research_proposal_semantic_drift.md). It is organized into four parts: (1) the generative and representation-learning foundations the project builds on, (2) instruction-based editing models and datasets, (3) consistency-preserving editing techniques — the closest prior art to this project's mitigation strategies, and (4) a synthesis identifying the specific gap this project fills. Source PDFs are in `../Research Papers - Existing/`.

---

## 1. Foundations

### 1.1 Latent Diffusion Models
**Rombach, Blattmann, Lorenz, Esser, Ommer — *High-Resolution Image Synthesis with Latent Diffusion Models* — CVPR 2022**

Diffusion models generate images by learning to reverse a noising process, but running that process directly in pixel space is computationally expensive — training can take hundreds of GPU-days. Rombach et al. separate image generation into two stages: (1) an autoencoder learns a lower-dimensional latent space that is perceptually equivalent to pixel space, and (2) a diffusion model is trained to generate in that latent space instead. This cuts computational cost sharply while preserving image quality, and their cross-attention conditioning mechanism (used to inject text or other conditioning into the U-Net) is what later work — including Prompt-to-Prompt and every editing method below — manipulates to control *what* the model generates and *where*. This is the backbone architecture Stable Diffusion, InstructPix2Pix, and nearly every other paper in this review are built on. It is why this project's Drift Score is model-agnostic in principle but only needs to be validated against Stable-Diffusion-family editors in the 30-day scope.

### 1.2 Vision-Language Similarity
**Radford, Kim, Hallacy, et al. — *Learning Transferable Visual Models From Natural Language Supervision (CLIP)* — ICML 2021**

CLIP jointly trains an image encoder and a text encoder on 400M (image, text) pairs using a contrastive objective: given a batch of N image-text pairs, the model learns to pull the correct pairings together in embedding space and push incorrect pairings apart. The result is a shared embedding space where cosine similarity between an image and a text (or between two images) is a meaningful, general-purpose similarity signal, transferable zero-shot to new tasks without fine-tuning. This is the exact mechanism the proposal's `compute_drift_score()` relies on: embedding a pre-edit and post-edit region crop and measuring `1 - cosine_similarity` as a proxy for "how much this region changed." It's worth being explicit in the write-up that CLIP similarity is a *proxy* for perceptual change, not a ground-truth measure — the paper itself does not evaluate CLIP for this fine-grained, region-level use case (its 30+ evaluation datasets are classification/retrieval-style), which is precisely why the proposal's optional human-perceptual check (Section 5.5) matters.

### 1.3 Promptable Segmentation
**Kirillov, Mintun, Ravi, et al. — *Segment Anything (SAM)* — ICCV 2023**

SAM introduces a promptable segmentation task, a model, and a data engine that together produce SA-1B, a dataset of 1B+ masks over 11M images. The model architecture is deliberately split into a (heavy) image encoder that runs once per image, and a (lightweight, ~50ms) prompt encoder + mask decoder that can be queried repeatedly with different prompts (points, boxes, or masks) without re-running the encoder. It generalizes zero-shot to new image distributions. For this project, SAM supplies the region decomposition that turns a raw pre/post image pair into a set of comparable region crops — without it, there is no non-target-region granularity to measure drift over, only whole-image similarity, which would blur out exactly the localized side effects this project is trying to catch. The encoder/decoder split is also a practical fit for the 30-day, Colab-GPU-limited budget in Section 6 of the proposal.

---

## 2. Instruction-Based Image Editing

### 2.1 InstructPix2Pix
**Brooks, Holynski, Efros — *InstructPix2Pix: Learning to Follow Image Editing Instructions* — CVPR 2023**

The central editor this project evaluates. Its key contribution is training data generation, not architecture: it combines a fine-tuned GPT-3 (to turn an input caption into an instruction + edited caption) with Stable Diffusion + Prompt-to-Prompt (to turn that pair of captions into a pair of before/after images), producing 454,445 synthetic training triplets (input image, instruction, output image). A conditional diffusion model is then trained on this generated data to map (image, instruction) → edited image directly in the forward pass — no per-example fine-tuning, mask, or inversion required at inference time. This is exactly why it's the right choice for this project: it is fast (one forward pass per edit, which matters for running 4-5-step chains over 50-80 images on free Colab GPU) and takes instructions rather than full image descriptions.

The paper is also explicit about a limitation directly relevant to this project's premise: since training pairs are generated by Prompt-to-Prompt applied to *generated* image pairs, the model inherits no explicit supervision for "leave everything else untouched" beyond whatever Prompt-to-Prompt's attention-injection approximately preserves in the synthetic data. There is no mechanism in the architecture itself that constrains edits to a region — consistency is an emergent, not enforced, property. That's the mechanistic reason semantic drift should be expected to occur and to compound.

### 2.2 MagicBrush (dataset)
**Zhang, Mo, Chen, Sun, Su — *MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing* — NeurIPS 2023, Datasets & Benchmarks track**

MagicBrush addresses a data-quality problem with InstructPix2Pix's approach: because its training triplets are entirely synthetic (generated captions + generated images), they contain "a high volume of noise." MagicBrush instead pays crowd workers to manually produce 10,388 real editing turns across 5,313 "sessions" using an interactive image-editing tool (DALL-E 2's editor), explicitly supporting **single-turn and multi-turn** editing (see their Figure 1 example: three sequential edits — "make background a county fair" → "have him a cowboy hat" → "change the shirt to plaid" — applied to one image). This is worth flagging directly: MagicBrush already frames "multi-turn" as a first-class editing scenario and fine-tunes InstructPix2Pix on it, showing human-rated improvement. But — this is the important distinction for the gap section below — their multi-turn evaluation measures whether *each requested edit* succeeded (does turn 2's output match what "have him a cowboy hat" should produce), not how much of the *rest* of the image drifted from what it was before turn 2. It is close prior art for the dataset-construction side of this project (their 4-category task taxonomy could sanity-check the object-level vs. global split in Section 5.1) but does not measure the phenomenon this project targets.

### 2.3 Emu Edit
**Sheynin, Polyak, Singer, et al. (Meta) — *Emu Edit: Precise Image Editing via Recognition and Generation Tasks* — CVPR 2024**

Emu Edit's diagnosis of InstructPix2Pix-style models is direct: they "often struggle to accurately interpret and execute" instructions, especially ones that deviate slightly from training distribution. Their fix is multi-task training across sixteen task types (region-based edits, free-form edits, and auxiliary computer-vision tasks like segmentation/detection) plus learned per-task embeddings that steer generation toward the correct edit type. Relevant here mainly as evidence that instruction-following accuracy and collateral-damage-avoidance are being treated as two separate axes in the field's own diagnosis — Emu Edit's contribution targets the former (did the right thing happen) more than the latter (did *only* the right thing happen), which is the axis this project isolates.

### 2.4 HIVE
**Zhang, Yang, Feng, et al. — *HIVE: Harnessing Human Feedback for Instructional Visual Editing* — CVPR 2024**

HIVE fine-tunes InstructPix2Pix-style models using a learned reward model trained on human rankings of candidate edits, in a pipeline analogous to RLHF for language models. Their own qualitative comparison (Figure 1) shows the baseline model understanding an instruction like "remove" or "change to blue" but failing to identify the *correct object* to apply it to — a failure mode adjacent to, but distinct from, drift: it's about mistargeting the edit, whereas semantic drift (as scoped in this project) is about unintended change in regions *outside* the (correctly or incorrectly) targeted one. Useful as a contrast case in the write-up: instruction-following quality and regional-preservation quality are both broken in current models, but by different mechanisms, and HIVE's human-feedback fix targets the former.

---

## 3. Consistency-Preserving Editing (closest prior art to the mitigation strategies)

This cluster of papers is the most important addition to the original proposal's related work — they are the field's existing attempts at exactly the problem this project measures, which sharpens both the gap statement and the mitigation-strategy design (Section 5.4).

### 3.1 Prompt-to-Prompt
**Hertz, Mokady, Tenenbaum, Aberman, Pritch, Cohen-Or — *Prompt-to-Prompt Image Editing with Cross-Attention Control* — arXiv 2022 / ICLR 2023**

Already in the proposal, but worth restating precisely: the paper's own framing of the problem is "an innate property of an editing technique is to preserve most of the original image, while in text-based models, even a small modification of the text prompt often leads to a completely different outcome" — i.e., this is literally the semantic-drift problem, stated as the motivation for their method, three years before this project. Their fix is to inject the *source* image's cross-attention maps into the *edited* image's generation process, so that the correspondence between words and spatial regions established in the original generation is preserved when a word changes ("cat" → "dog" keeps the same spatial layout because the attention map for that token position is reused). It requires no mask. It's the mechanism InstructPix2Pix's training-data generation itself depends on (Section 2.1 above), and it is the direct inspiration for the proposal's attention-restricted editing stretch goal.

### 3.2 Null-text Inversion
**Mokady, Hertz, Aberman, Pritch, Cohen-Or — *Null-text Inversion for Editing Real Images using Guided Diffusion Models* — CVPR 2023**

Prompt-to-Prompt's attention-injection trick only works on images the diffusion model generated itself, because it requires access to the generation trajectory. To apply it to *real* (user-provided) photos, you first have to invert the image — find the noise vector that would have produced it. Null-text Inversion shows that naive DDIM inversion breaks down under classifier-free guidance (the technique needed for meaningful text-guided edits), and fixes it by optimizing only the *unconditional* ("null") text embedding at each diffusion step to pull the reconstruction back toward the true image, while leaving the model weights and the conditional embedding untouched. This is largely orthogonal to this project's scope (InstructPix2Pix edits directly, without inversion) but explains a design constraint worth naming explicitly in the write-up: the reason InstructPix2Pix-style forward-pass editing is attractive for a 30-day project is precisely that it sidesteps this entire inversion problem that Prompt-to-Prompt-style attention control otherwise requires for real images.

### 3.3 Plug-and-Play Diffusion Features
**Tumanyan, Geyer, Bagon, Dekel — *Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation* — CVPR 2023**

A different lever on the same problem: instead of manipulating cross-attention (word-to-region binding), Plug-and-Play shows that the U-Net's intermediate **spatial features** and **self-attention** — extracted from the source image's generation/inversion trajectory — encode structure and layout with high spatial granularity, and injecting them into the target generation preserves structure while letting appearance/semantics change freely with the new prompt. The paper explicitly contrasts this with Prompt-to-Prompt: cross-attention only captures "rough regions at the object level," while their spatial-feature injection is more fine-grained and isn't restricted to word-aligned prompts. This is directly relevant to the region-locking mitigation strategy (Section 5.4.2 of the proposal) as an alternative, more sophisticated preservation mechanism than "recomposite pixels from the previous step" — worth a sentence in Discussion/Future Work as a stronger mitigation than what's in scope for 30 days.

### 3.4 MasaCtrl
**Cao, Wang, Qi, Shan, Qie, Zheng — *MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing* — ICCV 2023**

MasaCtrl's motivation section states the problem this entire project is about about as directly as any paper reviewed here: existing editing methods "either fail to achieve effective complex non-rigid editing while maintaining the overall textures and identity, or require time-consuming fine-tuning." Their method converts self-attention (which normally only looks within the image being generated) into *mutual* self-attention that queries the Key/Value features of the source image's generation process — so the edited image's content is generated by "querying" correlated content from the source. Critically, they identify a failure mode worth citing directly in the proposal's risk section: when foreground and background share similar patterns/colors, mutual self-attention "confuses" them, producing messy results — which they fix with a mask (extracted from cross-attention) that restricts foreground features to query only foreground, background-to-background. This mask-guided refinement is essentially the same idea as the proposal's masked-conditioning mitigation, arrived at independently for a different purpose (synthesis consistency rather than drift *measurement*), and is good evidence that SAM-quality masking of source/target regions is a load-bearing requirement, not a nice-to-have.

### 3.5 Imagic
**Kawar, Zada, Lang, et al. (Google Research) — *Imagic: Text-Based Real Image Editing with Diffusion Models* — CVPR 2023**

Imagic performs complex non-rigid edits (e.g., a standing dog sitting down) on a *single* real image via a three-step process: optimize a text embedding to reconstruct the input image, fine-tune the diffusion model around that embedding, then linearly interpolate between the optimized and target embeddings to generate the output. Its cost profile is the opposite of InstructPix2Pix's: per-image optimization and fine-tuning, which is far too slow for a 4-5-step chain over 50-80 images on free Colab (it's designed for one high-quality edit to one image, not iterative chains). Cited here mainly as the origin of TEdBench, one of the field's standard editing-quality benchmarks, and as the paper MasaCtrl explicitly positions itself against as the "tuning-required" alternative to tuning-free consistency control.

---

## 4. Synthesis: What's Actually Missing

Putting these together sharpens the proposal's gap statement (Section 2, "Gap this project addresses") beyond what was written before the papers were read closely:

- **Consistency *techniques* exist, but as generation-time interventions on a single edit, not as an evaluation metric, and not across a chain.** Prompt-to-Prompt, Null-text Inversion, Plug-and-Play, and MasaCtrl are all methods for *producing* a more consistent single edit — none of them *measure* how much unintended change remains, and none address what happens when the output of one edit becomes the input to the next. This project's Drift Score is a measurement instrument that could, in principle, be used to *evaluate* any of these four methods' actual before/after preservation, which none of their own papers do quantitatively at the region level (MasaCtrl and Plug-and-Play both use whole-image or coarse metrics, not a per-region non-target-area score).
- **MagicBrush has multi-turn data but doesn't isolate collateral change.** It's the closest thing to this project's dataset design, and its 4-turn example chains are good evidence the field considers 4-5-step chains a realistic usage pattern (validating Section 5.1's chain length). But its evaluation protocol asks "did turn *i*'s target region change correctly," never "how much did every *other* region change by turn *i*," and it has no notion of a cumulative, chain-position-indexed score — which is exactly RQ2 (H1: super-linear compounding).
- **No paper reviewed treats mask-guided region preservation as a *measurable, comparable* strategy.** MasaCtrl's mask-guided mutual self-attention is functionally close to "masked conditioning," but it's baked into their generation method, evaluated only by whether their own outputs look good — not compared against a no-mitigation baseline on a common drift metric across a chain, which is precisely RQ3.

This confirms the original gap statement holds up under closer reading, and gives the write-up (Section 26-28 of the timeline) concrete, specific papers to cite when explaining *why* the Drift Score is a novel instrument rather than a reinvention of consistency-editing methods: those methods are interventions; this project is a ruler.

---

## References

Brooks, T., Holynski, A., & Efros, A. A. (2023). InstructPix2Pix: Learning to Follow Image Editing Instructions. *CVPR 2023*.

Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., & Zheng, Y. (2023). MasaCtrl: Tuning-Free Mutual Self-Attention Control for Consistent Image Synthesis and Editing. *ICCV 2023*.

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., & Cohen-Or, D. (2022). Prompt-to-Prompt Image Editing with Cross Attention Control. *arXiv:2208.01626*.

Kawar, B., Zada, S., Lang, O., Tov, O., Chang, H., Dekel, T., Mosseri, I., & Irani, M. (2023). Imagic: Text-Based Real Image Editing with Diffusion Models. *CVPR 2023*.

Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. *ICCV 2023*.

Mokady, R., Hertz, A., Aberman, K., Pritch, Y., & Cohen-Or, D. (2023). Null-text Inversion for Editing Real Images using Guided Diffusion Models. *CVPR 2023*.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. *CVPR 2022*.

Sheynin, S., Polyak, A., Singer, U., Kirstain, Y., Zohar, A., Ashual, O., Parikh, D., & Taigman, Y. (2024). Emu Edit: Precise Image Editing via Recognition and Generation Tasks. *CVPR 2024*.

Tumanyan, N., Geyer, M., Bagon, S., & Dekel, T. (2023). Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation. *CVPR 2023*.

Zhang, K., Mo, L., Chen, W., Sun, H., & Su, Y. (2023). MagicBrush: A Manually Annotated Dataset for Instruction-Guided Image Editing. *NeurIPS 2023, Datasets and Benchmarks Track*.

Zhang, S., Yang, X., Feng, Y., et al. (2024). HIVE: Harnessing Human Feedback for Instructional Visual Editing. *CVPR 2024*.
