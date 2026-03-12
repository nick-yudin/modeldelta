# model-diff

## One-line pitch
CLI that shows **what changed inside** between two model checkpoints — weight deltas, spectral structure, layer drift.

## Problem
Everyone who fine-tunes asks: "What did my training actually change?" Currently the only answer is benchmark deltas. No standard tool shows:
- Which layers/modules changed most?
- Is the change low-rank (like LoRA) or distributed?
- What fraction of the delta is signal vs noise?
- How concentrated are the changes spectrally?

## Solution
```bash
model-diff Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct --report html
```

## v0 scope (strict)

### In v0
1. **Weight delta norms** — per-module Frobenius norm of ΔW
2. **SVD of deltas** — effective rank, top-k singular values, spectral decay
3. **Delta sparsity / concentration** — fraction of Frobenius norm in top-k SVs; DARE-style sparsity ratio
4. **Cosine similarity** — per-module alignment between old and new weights
5. **JSON + HTML report** — static HTML with heatmaps and SVD plots
6. **CLI** — `model-diff <model_a> <model_b> --output report.html`

### NOT in v0
- ~~Attention KL~~ — architecture-sensitive, hard to interpret, scope poison
- ~~Entropy diff~~ — requires generation, chat templates, tokenizer alignment
- ~~FIM / Riemannian distance~~ — requires data pass, expensive
- ~~Principal angles across methods~~ — v1 feature
- ~~HuggingFace Space for arbitrary uploads~~ — gated weights, GPU quota, unsupported archs
- ~~LoRA adapter diff~~ — different tensor names, merge logic needed
- ~~Cross-architecture comparison~~ — meaningless without permutation alignment

### Output formats
- **JSON** — machine-readable, all metrics
- **HTML** — static report with embedded plots (no JS framework, single file)
- **CLI text** — quick summary table

## v0 constraints

### Input contract
- Two HuggingFace checkpoint IDs or local paths
- **Same architecture** (same model family, same config)
- **Same tensor names** (same parameter shapes)
- **Same tokenizer family**
- Typical use: base→instruct, checkpoint_v1→checkpoint_v2, merge_A→merge_B

### Hardware
- CPU-only (no GPU needed) — all computation via safetensors streaming + torch SVD on CPU
- Never loads both full models simultaneously
- Streams tensor-by-tensor: load shard → compute delta → SVD → free → next
- Colab CPU runtime (0.07 units/hr), ~20 min per 7B pair

## Architecture (v0)
```
model_diff/
  __init__.py
  cli.py              # Click CLI entrypoint
  core/
    weight_diff.py     # Per-module delta norms, cosine similarity
    svd_analysis.py    # SVD decomposition, effective rank, concentration
  report/
    json_report.py     # JSON export
    html_report.py     # Single-file static HTML with embedded plots
  utils/
    model_loader.py    # Memory-efficient streaming loader (safetensors)
```

## First user
Fine-tuning engineer comparing base→instruct or two LoRA merges.
Primary question: **"what changed?"** (not "did it regress?").

## Distribution
- PyPI: `pip install modeldelta`
- GitHub: open source, MIT license
- Demo: precomputed HTML reports for popular pairs (Qwen base→instruct, Llama base→instruct)

## Competitors / related
- **mergekit** — model merging, not comparison
- **TransformerLens** — circuits, not diffing
- **PEFT** — adapter training, not analysis
- No direct competitor for "model weight diff report"

## Roadmap
- **v0**: weight-space diff (norms, SVD, cosine, sparsity) + CLI + HTML report
- **v1**: activation CKA on fixed prompt pack, entropy diff, LoRA adapter support
- **v2**: HF Space with precomputed demos, principal angle analysis, multi-checkpoint timeline
