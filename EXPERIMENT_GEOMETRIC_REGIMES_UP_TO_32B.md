# Experiment: Geometric Regimes of Post-Training up to 32B

## Goal

- Use `modeldelta` to test whether open-weight post-training methods fall into a small number of reproducible geometric regimes in weight space.
- Stay strictly in the scope that the tool can already support:
  - same-family checkpoint pairs
  - weight-space only
  - no prompt-based evaluation required
- Raise the model size ceiling from `<=30B` to `<=32B` to include:
  - `Qwen2.5-32B -> Qwen2.5-32B-Instruct`
  - `Qwen2.5-32B -> DeepSeek-R1-Distill-Qwen-32B`

## Why this experiment

- The weak version of the claim is useless:
  - "different training gives different weight changes"
- The useful version is narrower:
  - post-training methods may occupy a small number of stable geometric regimes
- If true, that is useful in three ways:
  - as an empirical map of open-model post-training
  - as a practical taxonomy for `modeldelta`
  - as a constraint for later theory work

## Primary question

- Do `instruct`, `specialization`, `preference/post-SFT`, and `reasoning distillation` form separable geometric regimes across model families?

## Secondary questions

- Is `reasoning distillation` a distinct regime, or just another specialization variant?
- Are lab-specific fine-tuning styles stronger than method-specific patterns?
- Do multi-stage pipelines (`Base -> SFT -> DPO -> Instruct`) show stage-specific footprints?

---

## Hypotheses

### H1. Instruct tuning occupies a relatively mild, family-stable regime

- Expected pattern:
  - small-to-moderate `frob_norm_relative`
  - very high `cosine_sim`
  - moderate `concentration_top_k`
  - low LayerNorm movement
- Expected examples:
  - Qwen: very surgical
  - Llama/Mistral: standard
  - Gemma: heavier but still not reasoning-distill-like

### H2. Domain specialization is not the same regime as instruct tuning

- Expected pattern:
  - higher `frob_norm_relative`
  - lower `cosine_sim`
  - broader layer participation
  - more full-rank deltas than instruct tuning
- Candidates:
  - `base -> coder`
  - `base -> math`

### H3. Reasoning distillation occupies a distinct late-MLP-concentrated regime

- Expected pattern:
  - very small global `frob_norm_relative`
  - extremely high `concentration_top_k`
  - changes concentrated in deep `MLP` tensors
  - minimal change in embeddings, norms, and early layers

### H4. Multi-stage post-training separates by stage

- Expected pattern:
  - `Base -> SFT` differs from `SFT -> DPO`
  - `DPO -> Instruct` differs again
  - one family with explicit stage checkpoints should show a cleaner transition than mixed public pairs

---

## Success criteria

- We can define at least `3` reproducible geometric regimes with low within-class variance and clear between-class separation.
- At least one regime is method-specific rather than family-specific.
- Reasoning distillation either:
  - replicates as a distinct regime across `>=3` pairs, or
  - clearly fails to replicate and is rejected as a one-off anomaly.

## Failure criteria

- Metrics overlap so heavily that classes collapse into "small change vs large change" only.
- Family identity explains most structure and method label explains little.
- Reasoning distillation does not separate from ordinary specialization after `>=4` validated pairs.

---

## Metrics to collect

For every pair:

- global:
  - `frob_norm_relative`
  - `cosine_sim`
  - `effective_rank`
  - `concentration_top_k`
  - `spectral_alpha`
- component-level:
  - attention mean relative change
  - MLP mean relative change
  - LayerNorm mean relative change
  - embedding mean relative change
  - `lm_head` relative change
- profile-level:
  - early / middle / late layer mean change
  - deepest 25% layer share of total delta energy
  - `mlp_to_attn_ratio`
  - flat vs U-shaped vs late-heavy depth profile

## Derived regime features

- `late_layer_share`
- `mlp_share`
- `norm_share`
- `embed_share`
- `head_share`
- `top20_concentration_mean`
- `top20_concentration_min`
- `top20_concentration_max`
- `rank_per_param` or normalized effective-rank proxy

---

## Model buckets and run list

All model IDs below were verified on Hugging Face on March 12, 2026.

### Bucket A. Instruct controls

Purpose:

- establish the baseline geometry of ordinary post-training
- estimate within-family size trends

Priority A1:

| Pair ID | Model A | Model B | Label |
|---|---|---|---|
| A1 | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B-Instruct` | instruct |
| A2 | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-3B-Instruct` | instruct |
| A3 | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-7B-Instruct` | instruct |
| A4 | `Qwen/Qwen2.5-14B` | `Qwen/Qwen2.5-14B-Instruct` | instruct |
| A5 | `Qwen/Qwen2.5-32B` | `Qwen/Qwen2.5-32B-Instruct` | instruct |
| A6 | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-3B-Instruct` | instruct |
| A7 | `meta-llama/Llama-3.1-8B` | `meta-llama/Llama-3.1-8B-Instruct` | instruct |
| A8 | `mistralai/Mistral-7B-v0.3` | `mistralai/Mistral-7B-Instruct-v0.3` | instruct |
| A9 | `google/gemma-2-9b` | `google/gemma-2-9b-it` | instruct |
| A10 | `google/gemma-2-27b` | `google/gemma-2-27b-it` | instruct |

### Bucket B. Explicit stage transitions

Purpose:

- isolate stage-specific geometry without guessing the training pipeline

Priority A2:

| Pair ID | Model A | Model B | Label |
|---|---|---|---|
| B1 | `allenai/OLMo-2-1124-7B` | `allenai/OLMo-2-1124-7B-SFT` | base_to_sft |
| B2 | `allenai/OLMo-2-1124-7B-SFT` | `allenai/OLMo-2-1124-7B-DPO` | sft_to_dpo |
| B3 | `allenai/OLMo-2-1124-7B-DPO` | `allenai/OLMo-2-1124-7B-Instruct` | dpo_to_instruct |
| B4 | `allenai/OLMo-2-1124-13B` | `allenai/OLMo-2-1124-13B-SFT` | base_to_sft |
| B5 | `allenai/OLMo-2-1124-13B-SFT` | `allenai/OLMo-2-1124-13B-DPO` | sft_to_dpo |
| B6 | `allenai/OLMo-2-1124-13B-DPO` | `allenai/OLMo-2-1124-13B-Instruct` | dpo_to_instruct |

### Bucket C. Domain specialization controls

Purpose:

- distinguish "reasoning distill" from ordinary domain specialization

Priority A3:

| Pair ID | Model A | Model B | Label |
|---|---|---|---|
| C1 | `Qwen/Qwen2.5-7B` | `Qwen/Qwen2.5-Coder-7B` | coder_specialization |
| C2 | `Qwen/Qwen2.5-Coder-7B` | `Qwen/Qwen2.5-Coder-7B-Instruct` | coder_instruct |
| C3 | `Qwen/Qwen2.5-14B` | `Qwen/Qwen2.5-Coder-14B` | coder_specialization |
| C4 | `Qwen/Qwen2.5-Coder-14B` | `Qwen/Qwen2.5-Coder-14B-Instruct` | coder_instruct |
| C5 | `Qwen/Qwen2.5-Math-1.5B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` | reasoning_distill |
| C6 | `Qwen/Qwen2.5-Math-7B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | reasoning_distill |

Note:

- `DeepSeek-R1-Distill-Qwen-1.5B` and `7B` explicitly list `Qwen2.5-Math-*` as their base models on the model card.
- These are cleaner than comparing distilled models to plain Qwen base.

### Bucket D. Reasoning distillation main test

Purpose:

- test whether reasoning distillation is a distinct regime across size and family

Priority A4:

| Pair ID | Model A | Model B | Label |
|---|---|---|---|
| D1 | `Qwen/Qwen2.5-14B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | reasoning_distill |
| D2 | `Qwen/Qwen2.5-32B` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | reasoning_distill |
| D3 | `meta-llama/Llama-3.1-8B` | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | reasoning_distill |

Note:

- The DeepSeek model card explicitly lists:
  - `Qwen2.5-14B` as the base for `DeepSeek-R1-Distill-Qwen-14B`
  - `Qwen2.5-32B` as the base for `DeepSeek-R1-Distill-Qwen-32B`
  - `Llama-3.1-8B` as the base for `DeepSeek-R1-Distill-Llama-8B`

### Bucket E. Exploratory reasoning-like controls

Purpose:

- optional bucket if Phase 1 patterns look real
- useful for separating "reasoning-tuned" from "teacher-distilled"

Priority B:

| Pair ID | Model A | Model B | Label |
|---|---|---|---|
| E1 | `Qwen/Qwen2.5-32B` | `Qwen/QwQ-32B` | reasoning_like_exploratory |

Use only if:

- Buckets A-D already show clear structure
- and you want one ambiguous but high-interest out-of-distribution point

---

## Recommended execution order

### Phase 1. Baseline map

Run:

- A1-A10
- B1-B6

Target:

- establish instruct and explicit stage baselines before touching headline claims

### Phase 2. Distillation test

Run:

- C5-C6
- D1-D3

Target:

- determine whether the "late MLP concentrated" reasoning-distill footprint survives replication

### Phase 3. Specialization controls

Run:

- C1-C4

Target:

- separate reasoning distillation from ordinary specialization

### Phase 4. Exploratory outlier

Run:

- E1

Target:

- test whether a reasoning-like model without the same explicit distillation label lands in the same regime

---

## Experimental procedure

### Step 1. Run pairwise comparisons

For every pair:

- run `modeldelta`
- export:
  - JSON
  - HTML
- store metadata:
  - pair id
  - family
  - size
  - relation label
  - source URLs
  - date run
  - tool version

### Step 2. Normalize module taxonomy

Map raw tensor names into shared groups:

- `embed`
- `attn_q`
- `attn_k`
- `attn_v`
- `attn_o`
- `mlp_gate`
- `mlp_up`
- `mlp_down`
- `norm`
- `lm_head`

This is necessary so Qwen, Llama, Gemma, Mistral, and OLMo are comparable.

### Step 3. Build pair-level regime table

For each pair, aggregate:

- overall metrics
- per-group metrics
- layer-depth summaries
- top changed tensors

Output:

- one row per pair
- one row per tensor
- one row per layer-group aggregate

### Step 4. Visualize class structure

Generate:

- scatter plots:
  - `frob_norm_relative` vs `concentration_top_k`
  - `cosine_sim` vs `concentration_top_k`
  - `late_layer_share` vs `mlp_share`
- heatmaps:
  - layer depth x module group
- family-colored and method-colored versions
- simple clustering:
  - hierarchical clustering on pair-level features
  - UMAP only as optional appendix, not core evidence

### Step 5. Test stability of the regimes

Check:

- within-label variance
- between-label separation
- family holdout:
  - train a simple classifier on pair-level features to predict label
  - leave one family out if sample count permits

Important:

- this is still descriptive work
- classifier is only a compact summary of separability, not the paper's main claim

---

## What we expect to find

### Expected regime 1. Surgical instruct tuning

- likely models:
  - Qwen instruct pairs
- expected signature:
  - very low `frob_norm_relative`
  - near-1 cosine
  - moderate concentration
  - tiny norm movement
  - no deep-late collapse

### Expected regime 2. Standard to heavy instruct tuning

- likely models:
  - Llama
  - Mistral
  - Gemma
  - possibly OLMo final stages
- expected signature:
  - larger `frob_norm_relative`
  - still high cosine
  - broader layer spread
  - more model-family variation

### Expected regime 3. Broad specialization / rewrite

- likely models:
  - `base -> coder`
- expected signature:
  - much larger delta magnitude
  - lower cosine
  - lower concentration
  - full-depth or broad-depth rewriting

### Expected regime 4. Reasoning distillation

- likely models:
  - DeepSeek-R1-Distill family
- expected signature:
  - very small global change
  - very high concentration
  - deep-layer heavy
  - MLP-gate dominant
  - embeddings and norms almost frozen

### Expected regime 5. Stage-specific post-training

- likely models:
  - OLMo-2 chain
- expected signature:
  - `Base -> SFT` and `SFT -> DPO` should not look identical
  - final stage may sharpen or redirect the footprint rather than only scale it

---

## What would count as interesting enough for a paper

### Strong outcome

- reasoning distillation replicates as a distinct regime across:
  - Qwen 1.5B
  - Qwen 7B
  - Qwen 14B
  - Qwen 32B
  - Llama 8B
- and remains clearly separated from:
  - instruct tuning
  - coder specialization
  - OLMo SFT/DPO stages

Possible framing:

- `Post-training methods occupy a small number of reproducible geometric regimes in weight space`
- with reasoning distillation as the sharpest regime

### Medium outcome

- no universal reasoning-distill regime
- but a clean taxonomy still appears:
  - surgical instruct
  - heavy instruct
  - specialization/rewrite
  - stage-specific DPO/RL preference tuning

Possible framing:

- tool paper + empirical map

### Weak outcome

- patterns reduce to:
  - small update
  - large update
- family dominates everything
- no stable label-specific structure

Conclusion:

- useful for product diagnostics
- weak for standalone arXiv paper

---

## Go / no-go checkpoints

### Checkpoint 1

Run:

- A1-A10
- B1-B3

Go if:

- instruct pairs already show family-stable structure beyond plain magnitude
- OLMo stages show visible stage differences

Stop if:

- all separation collapses into one scalar size-of-delta axis

### Checkpoint 2

Run:

- C5-C6
- D1-D3

Go if:

- at least `4/5` reasoning-distill pairs cluster together by:
  - high concentration
  - late-layer dominance
  - MLP-heavy footprint

Stop if:

- reasoning-distill points overlap broadly with ordinary instruct or specialization pairs

### Checkpoint 3

Run:

- C1-C4

Go if:

- specialization controls clearly differ from reasoning distill

Stop if:

- coder and reasoning-distill are geometrically indistinguishable after normalization

---

## Practical notes

- Keep the paper question descriptive and narrow.
- Do not add benchmark evaluation in this phase.
- Do not add activation-space work in this phase.
- Do not use cross-generation pairs.
- Do not compare models with unclear lineage as core evidence.

## Deliverables

- `results/pairs.csv`
- `results/modules.parquet`
- `results/pair_features.csv`
- `figures/`
  - regime scatter plots
  - family-vs-method plots
  - layer heatmaps
  - OLMo stage transition plots
- short memo:
  - what replicated
  - what failed
  - whether the paper track is justified

---

## Sources used to verify model availability

- Qwen 2.5 base/instruct family:
  - https://huggingface.co/Qwen/Qwen2.5-7B
  - https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
  - https://huggingface.co/Qwen/Qwen2.5-32B
  - https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- Qwen Math / Coder:
  - https://huggingface.co/Qwen/Qwen2.5-Math-1.5B
  - https://huggingface.co/Qwen/Qwen2.5-Math-7B
  - https://huggingface.co/Qwen/Qwen2.5-Coder-7B
  - https://huggingface.co/Qwen/Qwen2.5-Coder-14B
  - https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
  - https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct
- DeepSeek distills:
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
  - https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B
- Meta, Mistral, Gemma, OLMo:
  - https://huggingface.co/meta-llama/Llama-3.1-8B
  - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
  - https://huggingface.co/meta-llama/Llama-3.2-3B
  - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
  - https://huggingface.co/mistralai/Mistral-7B-v0.3
  - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
  - https://huggingface.co/google/gemma-2-9b
  - https://huggingface.co/google/gemma-2-9b-it
  - https://huggingface.co/google/gemma-2-27b
  - https://huggingface.co/google/gemma-2-27b-it
  - https://huggingface.co/allenai/OLMo-2-1124-7B
  - https://huggingface.co/allenai/OLMo-2-1124-7B-SFT
  - https://huggingface.co/allenai/OLMo-2-1124-7B-DPO
  - https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct
  - https://huggingface.co/allenai/OLMo-2-1124-13B
  - https://huggingface.co/allenai/OLMo-2-1124-13B-SFT
  - https://huggingface.co/allenai/OLMo-2-1124-13B-DPO
  - https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct
