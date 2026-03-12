# model-diff — Status

## Current state: v0.1 complete with diagnostics, platform architecture defined

### What works
- **CLI package** (`model_diff/`) with `model-diff <A> <B>` entrypoint
- **Weight diff engine**: per-module Frobenius norm, cosine similarity (clamped), SVD analysis
- **Randomized SVD** for large matrices (>8192) via QR projection
- **Memory-efficient streaming**: loads one tensor at a time via safetensors, never both full models
- **3 output formats**: JSON, HTML (single file, embedded base64 plots), CLI text
- **Diagnostic conclusions**: automatic human-readable "diagnosis" in all formats
- **Database schema**: HF Dataset schema for shared results (pairs + modules tables)
- **3 Colab notebooks**: Qwen POC, Llama POC, batch runner

### Batch results (7 pairs, completed)
Drive path: `~/Library/CloudStorage/GoogleDrive-n.yudin@gmail.com/My Drive/model_diff/`

| Pair | Profile | ΔW/W | eff_rank | conc | cos_sim |
|------|---------|------|----------|------|---------|
| Qwen2.5-3B → Instruct | SURGICAL | 0.009 | 1209 | 0.214 | 0.99995 |
| Qwen2.5-7B → Instruct | SURGICAL | 0.012 | 2166 | 0.141 | 0.99996 |
| Qwen2.5-14B → Instruct | SURGICAL | 0.008 | 3031 | 0.143 | 0.99998 |
| Llama-3.1-8B → Instruct | STANDARD | 0.035 | 2579 | 0.103 | 0.99978 |
| Llama-3.2-3B → Instruct | HEAVY | 0.115 | 2076 | 0.072 | 0.99273 |
| Mistral-7B-v0.3 → Instruct | STANDARD | 0.016 | 2562 | 0.140 | 0.99995 |
| Gemma-2-9B → IT | EXTREME | 0.148 | 2799 | 0.046 | 0.98538 |

### Key findings
- **Qwen**: Most conservative SFT. LoRA-like structure (conc 0.14-0.21). Attention ≈ MLP balanced.
- **Llama-3.1**: Standard SFT. MLP > Attention. lm_head dominates.
- **Llama-3.2**: Aggressive SFT — full-rank changes, LayerNorm modified (0.021).
- **Mistral**: Clean SFT, LayerNorm essentially frozen (4e-6). MLP > Attention.
- **Gemma-2**: Most extreme — full-rank rewriting (conc 4.6%), MLP gate_proj dominates top changes.
- **Universal**: lm_head always among most changed. LayerNorm nearly untouched (except Llama-3.2, Gemma).

### Files
```
model-diff/
├── STATUS.md                  # ← this file
├── pyproject.toml              # pip installable, Click CLI
├── model_diff/
│   ├── __init__.py             # version = "0.1.0"
│   ├── cli.py                  # Click entrypoint: compare command
│   ├── core/
│   │   └── weight_diff.py      # analyze_delta(), compare_models()
│   ├── report/
│   │   ├── diagnostics.py      # NEW: diagnostic conclusions engine
│   │   ├── html_report.py      # Single-file HTML with heatmaps + SVD plots + diagnostics
│   │   ├── json_report.py      # JSON export + diagnostics
│   │   └── text_report.py      # Terminal table + diagnostics
│   ├── database/
│   │   ├── __init__.py
│   │   └── schema.py           # NEW: HF Dataset schema (pairs + modules tables)
│   └── utils/
│       └── model_loader.py     # HF download, safetensors streaming
└── notebooks/
    ├── model_diff_v0_poc_colab.ipynb
    ├── model_diff_v0_llama_colab.ipynb
    └── model_diff_v0_batch_colab.ipynb
```

## Platform architecture

### Unified platform (4 access modes)

```
                    ┌─────────────────────────────┐
                    │   HF Dataset: model-diff-db  │
                    │   (pairs + modules tables)   │
                    └──────────┬──────────────────┘
                               │
         ┌─────────┬───────────┼───────────┬──────────┐
         │         │           │           │          │
    ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼────┐ ┌──▼───┐
    │  CLI   │ │ Space  │ │ Colab  │ │  API   │ │Agent │
    │ local  │ │ ≤3B    │ │ >3B    │ │ query  │ │ MCP  │
    └────────┘ └────────┘ └────────┘ └────────┘ └──────┘
```

| Component | Description | Cost |
|-----------|-------------|------|
| **CLI** | `pip install modeldelta` — local computation, any size | Free |
| **HF Space** | Gradio app, CPU, models ≤3B + precomputed gallery | Free (community) |
| **Colab** | Auto-generated notebook for >3B models | Free (user's Colab) |
| **HF Dataset** | Shared results DB, queryable via `datasets` API | Free |
| **Agent API** | Structured JSON schema, searchable by model name | Free |

### Data flow
1. User computes pair (CLI / Space / Colab)
2. Results (JSON) → push to HF Dataset
3. Anyone can browse/query database
4. HTML reports include diagnostic "diagnosis"
5. Agents discover tool via model card + structured README

### Agent-friendly design
- **Model card** with structured capabilities, input/output schema
- **README** with `tool_use` JSON schema for AI agents
- **HF Dataset** searchable by model name → agents can answer "how was X fine-tuned?"
- **JSON output** includes `diagnostics.profile_tag` + `diagnostics.summary` — agents can parse and relay

## Tracks

### Track 1: CLI + diagnostics (v0.1) — DONE
- [x] CLI working locally
- [x] Memory-optimized SVD (peak ~6.6 GB for 7B)
- [x] 3 output formats (JSON, HTML, text)
- [x] Diagnostic conclusions in all formats
- [x] Database schema for shared results

### Track 2: Distribution
- [ ] GitHub repo + PyPI release
- [ ] README.md with usage examples + screenshots
- [ ] Push 7 batch results to HF Dataset
- [ ] HF Space (precomputed gallery + ≤3B online computation)
- [ ] Reddit r/LocalLLaMA + Twitter launch

### Track 3: Paper data
- [x] 7-pair batch run completed
- [ ] LoRA vs full SFT comparison
- [ ] "Fingerprint": DPO vs SFT vs RLHF spectral signatures
- [ ] Scale to 50+ modifications of one base

### Design decisions
- **v0 is weight-space only** — no activations, no prompts needed
- **Input contract**: same architecture, same tensor names, same shapes
- **First user**: fine-tuning engineer. Question: "what changed?"
- **Diagnostics calibrated** on 7 base→instruct pairs (Qwen, Llama, Mistral, Gemma)

### Runtime
- **Standard CPU** (0.07/hr) — memory-optimized SVD
- Peak RAM ~6.6 GB for 7B lm_head
- ~18 min per 7B pair including download
- GPU not used
