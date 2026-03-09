"""HF Dataset schema for modeldelta shared results database.

Dataset: modeldelta-results (HuggingFace Hub)

Two tables:
  1. `pairs` — one row per model pair comparison (summary-level)
  2. `modules` — one row per module per pair (detail-level, linked by pair_id)

Usage:
  - CLI/Colab/Space push results after computation
  - Anyone can query/download via `datasets` library or HF API
  - Agents can search by model name, profile tag, etc.
"""

from __future__ import annotations


# ── Pairs table schema ──────────────────────────────────

PAIRS_FEATURES = {
    # Identity
    "pair_id": "string",           # SHA256(model_a + model_b + version)
    "model_a": "string",           # HF model ID (e.g. "Qwen/Qwen2.5-7B")
    "model_b": "string",           # HF model ID (e.g. "Qwen/Qwen2.5-7B-Instruct")
    "model_family": "string",      # e.g. "qwen2.5", "llama3.1"
    "model_size_b": "float32",     # Approximate size in billions of parameters

    # Summary metrics
    "n_tensors": "int32",
    "n_skipped": "int32",
    "total_frob_norm": "float32",
    "mean_frob_relative": "float32",
    "mean_cosine_sim": "float32",
    "mean_effective_rank": "float32",
    "mean_concentration": "float32",
    "mean_sparsity": "float32",

    # Diagnostics
    "profile_tag": "string",       # surgical / standard / heavy / extreme
    "diagnosis_summary": "string", # Human-readable paragraph

    # Metadata
    "modeldelta_version": "string",
    "computed_at": "string",       # ISO timestamp
    "computed_by": "string",       # "cli" / "colab" / "space"
    "top_k": "int32",              # SVD top-k parameter used
}


# ── Modules table schema ────────────────────────────────

MODULES_FEATURES = {
    "pair_id": "string",           # FK to pairs table
    "name": "string",              # e.g. "model.layers.0.self_attn.v_proj.weight"
    "n_params": "int64",
    "shape": "string",             # JSON list, e.g. "[4096, 4096]"

    # Metrics
    "frob_norm": "float32",
    "frob_norm_relative": "float32",
    "cosine_sim": "float32",
    "sparsity": "float32",

    # SVD metrics (nullable)
    "has_svd": "bool",
    "effective_rank": "float32",
    "concentration_top_k": "float32",
    "spectral_alpha": "float32",
    "n_singular_values": "int32",
    # top_singular_values stored as JSON string (list of floats)
    "top_singular_values": "string",
}


def make_pair_id(model_a: str, model_b: str, version: str = "v0") -> str:
    """Deterministic pair ID."""
    import hashlib
    key = f"{model_a}|{model_b}|{version}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def results_to_pair_row(
    modules: list[dict],
    model_a: str,
    model_b: str,
    diagnosis_summary: str = "",
    profile_tag: str = "",
    model_family: str = "",
    model_size_b: float = 0.0,
    computed_by: str = "cli",
    top_k: int = 20,
) -> dict:
    """Convert analysis results to a pairs table row."""
    import math
    from datetime import datetime, timezone
    import modeldelta

    svd_mods = [m for m in modules if m.get("has_svd")]

    def _safe_mean(vals):
        return sum(vals) / max(1, len(vals)) if vals else 0.0

    return {
        "pair_id": make_pair_id(model_a, model_b),
        "model_a": model_a,
        "model_b": model_b,
        "model_family": model_family,
        "model_size_b": model_size_b,
        "n_tensors": len(modules),
        "n_skipped": 0,
        "total_frob_norm": math.sqrt(sum(m["frob_norm"] ** 2 for m in modules)),
        "mean_frob_relative": _safe_mean([m["frob_norm_relative"] for m in modules]),
        "mean_cosine_sim": _safe_mean([min(1.0, m["cosine_sim"]) for m in modules]),
        "mean_effective_rank": _safe_mean(
            [m["effective_rank"] for m in svd_mods if m.get("effective_rank")]
        ),
        "mean_concentration": _safe_mean(
            [m["concentration_top_k"] for m in svd_mods if m.get("concentration_top_k")]
        ),
        "mean_sparsity": _safe_mean([m["sparsity"] for m in modules]),
        "profile_tag": profile_tag,
        "diagnosis_summary": diagnosis_summary,
        "modeldelta_version": modeldelta.__version__,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "computed_by": computed_by,
        "top_k": top_k,
    }


def results_to_module_rows(
    modules: list[dict],
    model_a: str,
    model_b: str,
) -> list[dict]:
    """Convert analysis results to module table rows."""
    import json as _json
    pair_id = make_pair_id(model_a, model_b)
    rows = []
    for m in modules:
        rows.append({
            "pair_id": pair_id,
            "name": m["name"],
            "n_params": m["n_params"],
            "shape": _json.dumps(m["shape"]),
            "frob_norm": m["frob_norm"],
            "frob_norm_relative": m["frob_norm_relative"],
            "cosine_sim": min(1.0, m["cosine_sim"]),
            "sparsity": m["sparsity"],
            "has_svd": m.get("has_svd", False),
            "effective_rank": m.get("effective_rank", 0.0) or 0.0,
            "concentration_top_k": m.get("concentration_top_k", 0.0) or 0.0,
            "spectral_alpha": m.get("spectral_alpha", 0.0) or 0.0,
            "n_singular_values": m.get("n_singular_values", 0) or 0,
            "top_singular_values": _json.dumps(m.get("top_singular_values", [])),
        })
    return rows
