# The Weight-Space O/Q Ratio

**Paper:** *The Weight-Space O/Q Ratio: A Robust Fingerprint of LLM Post-Training Regimes and an R1-Distillation Scaling Pattern*

## Summary

The O/Q ratio — the ratio of mean relative Frobenius norms of weight deltas in the output projection (W_O) vs. the query projection (W_Q) — is a lightweight, architecture-portable metric for characterizing LLM post-training from weights alone. No forward passes, no training data, no GPU required.

**Key finding:** O/Q > 2 reliably identifies R1-style distillation across 59 model pairs and 10 architecture families, with zero false positives among 53 non-R1 pairs. R1-distill O/Q follows a log-linear trend with model size (1.99 at 1.5B → 7.89 at 70B), while standard post-training stays flat near 1.0.

## Contents

| File | Description |
|------|-------------|
| `oq_ratio_paper.pdf` | Full paper (16 pages) |
| `references.bib` | Bibliography |
| `final_table_paper.csv` | All 59 model pairs with O/Q values and metadata |
| `figures/` | All paper figures (pipeline, attention block, taxonomy, scaling, null distributions, per-layer scatter) |

## Reproducing

All model pair comparisons can be reproduced using the `modeldelta` CLI:

```bash
pip install modeldelta
modeldelta compare <base_model> <finetuned_model> -o result.json
```

See the [main README](../README.md) for full documentation.

## Citation

```bibtex
@article{yudin2026oqratio,
  title={The Weight-Space O/Q Ratio: A Robust Fingerprint of LLM Post-Training Regimes and an R1-Distillation Scaling Pattern},
  author={Yudin, Nikolay},
  year={2026}
}
```
