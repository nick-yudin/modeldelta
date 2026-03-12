# Expert Briefing: modeldelta — Can This Become an arXiv Paper?

**Context:** We built a tool, have 11 preliminary model comparisons, and found one potentially interesting signal. We want an honest expert opinion: is there a paper here, and if so, what is the right framing?

---

## 1. What the Tool Does

**modeldelta** compares two HuggingFace checkpoints (e.g. base → instruct) tensor-by-tensor, streaming safetensors one layer at a time — no GPU, no full model load.

**For every weight matrix W_a → W_b it computes:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `frob_norm_relative` | ‖W_b − W_a‖_F / ‖W_a‖_F | Relative magnitude of change |
| `cosine_sim` | cos(W_a.flat, W_b.flat) | Direction preservation |
| `effective_rank` | exp(H(σ/‖σ‖₁)) | Shannon entropy over singular values of ΔW |
| `concentration_top_k` | ‖σ[:20]‖² / ‖σ‖² | Fraction of ΔW energy in top-20 singular vectors |
| `spectral_alpha` | power-law fit of log σ vs log rank | Decay rate of singular value spectrum |

**Key design choice:** SVD is computed on the **delta matrix** ΔW = W_b − W_a, not on W_b. This reveals the structure of what training *added*, not what the model *is*.

**Diagnostic profiles** based on mean_frob across all tensors:
- `SURGICAL`: < 0.015
- `STANDARD`: 0.015–0.05
- `HEAVY`: 0.05–0.12
- `EXTREME`: > 0.12

Secondary diagnostics: attention vs MLP ratio, LayerNorm change, U-shaped vs flat layer profile.

---

## 2. All 11 Results

| Profile | Model A | Model B | frob | cos | eff_rank | conc_top20 |
|---------|---------|---------|------|-----|----------|------------|
| EXTREME | Qwen2.5-7B | **Qwen2.5-Coder-7B** | 0.862 | 0.605 | 2366 | 0.065 |
| EXTREME | gemma-2-9b | gemma-2-9b-it | 0.148 | 0.985 | 2799 | 0.046 |
| HEAVY | Llama-3.2-3B | Llama-3.2-3B-Instruct | 0.115 | 0.993 | 2076 | 0.072 |
| HEAVY | SmolLM2-1.7B | SmolLM2-1.7B-Instruct | 0.051 | 0.999 | 1611 | 0.093 |
| STANDARD | Llama-3.1-8B | Llama-3.1-8B-Instruct | 0.035 | 0.9998 | 2579 | 0.103 |
| STANDARD | Mistral-7B-v0.3 | Mistral-7B-Instruct-v0.3 | 0.016 | 0.9999 | 2562 | 0.140 |
| SURGICAL | Qwen2.5-7B | Qwen2.5-7B-Instruct | 0.012 | 0.99996 | 2166 | 0.141 |
| SURGICAL | Qwen2.5-3B | Qwen2.5-3B-Instruct | 0.0086 | 0.99995 | 1209 | 0.214 |
| SURGICAL | Qwen2.5-1.5B | Qwen2.5-1.5B-Instruct | 0.0087 | 0.99995 | 938 | 0.222 |
| SURGICAL | Qwen2.5-14B | Qwen2.5-14B-Instruct | 0.0081 | 0.99998 | 3031 | 0.143 |
| SURGICAL | Qwen3.5-27B | **Qwen3.5-27B-Claude-Reasoning-Distilled** | 0.0030 | 0.99999 | 933 | **0.918** |

### Per-component breakdown (for pairs with full data):

**Qwen2.5-7B → Coder-7B:**
- attn=0.914, mlp=1.200, embed=1.285, norm=0.221
- conc: mean=0.065, min=0.014, max=0.280 — full-rank, no structure
- cosine=0.605: the model moved to a completely different region of weight space

**Qwen2.5-1.5B → Instruct (representative SURGICAL):**
- attn=0.0097, mlp=0.0125, norm=0.0001
- conc: mean=0.222 — some low-rank structure, but not extreme
- LayerNorm essentially frozen

**Qwen3.5-27B → Reasoning Distilled:**
- attn=0.0018, **mlp=0.0073**, norm=0.0000, embed=0.0000
- **conc: mean=0.918, min=0.807, max=0.985** — every single tensor
- Changes concentrated in MLP gate_proj, layers 34–62 (deep layers only)
- Early layers, embeddings, norms: frob ≈ 0.000

---

## 3. Patterns Found

### Pattern A — Qwen uses a radically more conservative SFT recipe
Qwen2.5 {1.5B, 3B, 7B, 14B} → Instruct: frob = 0.008–0.012.
Llama-3.1-8B → Instruct: frob = 0.035 (3× more).
Gemma-2-9b → it: frob = 0.148 (15× more).

Same task (base→instruct alignment), order-of-magnitude difference in weight change. Either Qwen uses LoRA, very low LR, or their base already incorporates much of the instruction-following capability.

### Pattern B — Coder-7B is not a fine-tune; it is a different model
Qwen2.5-7B → Coder-7B: frob=0.862, cosine=0.605.
Qwen2.5-7B → Instruct: frob=0.012, cosine=0.99996.

The "Coder" model shares architecture and was probably initialized from the same base, then trained on a completely different data mixture. The weight-space distance is orders of magnitude larger than any fine-tuning. The tool makes this immediately obvious without running a single benchmark.

### Pattern C — Gemma-2's training scaled weights without rotating them
frob=0.148 (EXTREME magnitude) but cosine=0.985 (high direction preservation).
The instruct training *amplified* Gemma's weights rather than redirecting them.
This is mechanistically different from Coder-7B (new directions) and Qwen (minimal change).

### Pattern D — Reasoning distillation has a unique geometric signature ← most interesting

**Standard SFT/RLHF** (all other pairs): concentration_top20 = 0.046–0.222.
**Reasoning distillation** (Claude Opus traces → Qwen3.5-27B): concentration_top20 = **0.918**.

- Min per-tensor concentration = 0.807; max = 0.985
- Every single MLP tensor independently shows this property
- Changes are almost entirely MLP gate_proj, deep layers (34–62 of 62)
- Embedding, attention, LayerNorm, early layers: Frobenius norm ≈ 0.000

**Interpretation attempt:** The delta ΔW for each affected MLP layer lives in approximately a 1–2 dimensional subspace. Total weight change is tiny (frob=0.003), but the few dimensions that did change are extremely concentrated. This is structurally different from LoRA (which also produces low-rank deltas but doesn't zero out early layers and embeddings entirely).

---

## 4. What We Can Easily Run Next

The tool runs CPU-only in ~40 min for 7B models (Google Colab, free tier). Within 1 week we could add:

**Reasoning-distilled models (to test Pattern D at scale):**
- DeepSeek-R1-Distill-Qwen-{1.5B, 7B, 14B, 32B}
- DeepSeek-R1-Distill-Llama-{8B, 70B}
- QwQ-32B vs Qwen2.5-32B (base)
- Sky-T1-32B vs Qwen2.5-32B
- Any other R1-style distilled model

**More base→instruct pairs (to characterize the space):**
- Qwen2.5-{0.5B, 32B, 72B}
- Phi-3-mini → Phi-3-mini-instruct
- OLMo-2-7B → OLMo-2-7B-instruct
- Gemma-3 family

**Version comparisons:**
- Llama-2-7B → Llama-3-8B (cross-generation)
- Mistral-7B-v0.1 → v0.2 → v0.3

---

## 5. What We Cannot Easily Add

- **Benchmark correlations**: no capability measurements paired with these pairs
- **Intermediate checkpoints**: not public for most models
- **Training details**: LR, data size, steps — not disclosed
- **LoRA adapter inspection**: we compare merged weights only

---

## 6. Honest Self-Assessment

**Strongest signal:** concentration_top20 = 0.918 for reasoning distillation vs 0.046–0.222 for all SFT/RLHF. This is a 4–20× difference on a geometric property of the delta matrix, and it's consistent across every single tensor in the pair.

**Weakest point:** One data point. We have exactly one reasoning-distilled model. We don't know if this is a property of *distillation* in general, of *reasoning distillation* specifically, of *this particular model* (Qwen3.5-27B architecture), or of the specific teacher (Claude Opus traces).

**Our honest doubt:** Is this finding interesting, or is it obvious in retrospect? "Knowledge distillation produces more structured weight changes than SFT" might be expected from the KD literature — you're training to match a specific output distribution, which could naturally produce lower-rank updates.

---

## 7. Questions for the Expert

1. **Is concentration_top20 of ΔW a known metric?** Has anyone published weight-space analysis of SFT vs distillation at the per-tensor level? If so, we'd need to position carefully.

2. **Is Pattern D (reasoning distillation signature) worth pursuing?** If we run 10–15 reasoning-distilled models and show they all have concentration > 0.7 while SFT is always < 0.25, is that publishable, or does it need a *consequence* (e.g., a prediction about behavior)?

3. **Pattern A (Qwen conservatism) vs Pattern D (distillation geometry)** — which is more interesting scientifically?

4. **Is a descriptive taxonomy paper viable at arXiv?** N=50 pairs, characterize the fine-tuning landscape by weight-space geometry. No causal claim, just a map. Would that pass?

5. **Missing causal chain** — do you see a natural experiment that connects weight-space geometry to something measurable? For example: does high concentration predict better retention of base model knowledge? Predict OOD generalization? Predict susceptibility to jailbreak?

6. **Mechanistic interpretability angle**: do the top-20 singular directions of the distillation ΔW correspond to interpretable features? Is that a feasible follow-up?

---

## Links

- Gallery + results: https://nick-yudin.github.io/modeldelta/
- HF Space (online ≤3B): https://huggingface.co/spaces/NikolayYudin/modeldelta
- Results JSON: https://huggingface.co/datasets/NikolayYudin/modeldelta-results
- Code: https://github.com/nick-yudin/modeldelta
- PyPI: `pip install modeldelta`
