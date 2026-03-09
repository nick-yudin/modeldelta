# modeldelta

**See what changed inside any model checkpoint.** Weight deltas, SVD structure, spectral analysis, and diagnostic conclusions — in one command.

```bash
pip install modeldelta
modeldelta Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct -o report.html
```

**[Live demo & precomputed reports →](https://nick-yudin.github.io/modeldelta/)**

## What it does

Compares two model checkpoints (base vs instruct, v1 vs v2, merge A vs merge B) and produces:

- **Per-module metrics**: Frobenius norm of weight delta, cosine similarity, sparsity
- **SVD analysis**: effective rank, top-k singular value concentration, spectral decay
- **Layer heatmaps**: 4 metrics across all layers and module types
- **Diagnostic conclusions**: human-readable "diagnosis" — was this surgical SFT or heavy rewriting?

## Output formats

| Format | Flag | Description |
|--------|------|-------------|
| Terminal | (default) | Quick summary table with diagnostics |
| HTML | `-o report.html` | Single-file report with embedded plots, heatmaps, SVD spectra, and diagnostic summary |
| JSON | `-o report.json` | Machine-readable, includes `diagnostics.profile_tag` and `diagnostics.summary` |

## Example output (terminal)

```
modeldelta: Qwen/Qwen2.5-7B → Qwen/Qwen2.5-7B-Instruct
Tensors: 283 analyzed, 85 skipped
Total ||ΔW||: 25.44  |  Mean cos_sim: 0.99996  |  Mean eff_rank: 2166

Module                                                    ΔW/W   cos_sim  eff_rank    conc  spars
───────────────────────────────────────────────────────────────────────────────────────────────────
lm_head.weight                                          0.0395  0.99990      1445   0.540  0.002
model.layers.0.self_attn.v_proj.weight                  0.0279  0.99998      1839   0.178  0.002
...

─── Diagnosis ───

Qwen/Qwen2.5-7B-Instruct was surgically fine-tuned with minimal weight changes.
Mean relative change: 0.0120, cosine similarity: 0.99996, total ||ΔW||: 25.44.

  • Surgical fine-tuning
    Mean relative change is very small (0.0120). Typical of careful SFT.

  ▸ Output head is the most changed module
    lm_head change (0.0395) is 3.3× the body average.

  • LayerNorm weights nearly untouched
    LayerNorm mean change is 0.000088 — essentially frozen.
```

## Diagnostic profiles

The diagnostic engine classifies fine-tuning into four profiles based on 7 calibration pairs:

| Profile | Mean ΔW/W | Example |
|---------|-----------|---------|
| **SURGICAL** | < 0.015 | Qwen2.5 family — minimal, targeted changes |
| **STANDARD** | 0.015–0.05 | Llama-3.1-8B, Mistral-7B — typical SFT |
| **HEAVY** | 0.05–0.12 | Llama-3.2-3B — aggressive training, LayerNorm modified |
| **EXTREME** | > 0.12 | Gemma-2-9B — full-rank rewriting, possible continued pre-training |

## Requirements

- Python >= 3.9
- CPU only — no GPU needed
- ~6.6 GB peak RAM for 7B models (memory-optimized SVD)
- ~18 minutes per 7B pair including download

## CLI options

```
modeldelta MODEL_A MODEL_B [OPTIONS]

  MODEL_A, MODEL_B    HuggingFace model IDs or local paths

Options:
  -o, --output PATH   Output file (.json or .html). Omit for terminal text.
  --top-k INT         Number of top singular values to track [default: 20]
  --top-n INT         Number of modules to show in text output [default: 20]
  --token TEXT        HuggingFace token (or set HF_TOKEN env var)
```

## For AI agents

modeldelta produces structured JSON output suitable for programmatic use:

```json
{
  "model_a": "Qwen/Qwen2.5-7B",
  "model_b": "Qwen/Qwen2.5-7B-Instruct",
  "n_tensors": 283,
  "diagnostics": {
    "profile_tag": "surgical",
    "summary": "Qwen2.5-7B-Instruct was surgically fine-tuned...",
    "findings": [
      {
        "category": "magnitude",
        "severity": "info",
        "title": "Surgical fine-tuning",
        "detail": "Mean relative change is very small (0.0120)..."
      }
    ]
  },
  "modules": [...]
}
```

**Use cases for agents:**
- "How was model X fine-tuned?" → run modeldelta, read `diagnostics.summary`
- "Which fine-tune should I pick?" → compare profile_tags across variants
- "Did this merge break anything?" → check for unusual patterns (EXTREME profile, LayerNorm modified)

## How it works

1. Downloads safetensors files (not full model) via `huggingface_hub`
2. Streams tensor-by-tensor: load → compute delta → SVD → free → next
3. Never loads both full models simultaneously
4. Memory-optimized SVD: in-place delta computation, free inputs before SVD phase
5. Randomized SVD via QR projection for matrices > 8192

## License

MIT
