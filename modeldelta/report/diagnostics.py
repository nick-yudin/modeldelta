"""Diagnostic conclusions engine — human-readable interpretations of weight diff metrics.

Calibrated against 7 base→instruct pairs:
  Qwen2.5-{3B,7B,14B}, Llama-{3.1-8B,3.2-3B}, Mistral-7B-v0.3, Gemma-2-9B
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# ── Thresholds (calibrated on 7 pairs) ──────────────────────────

# Overall change magnitude (mean ||ΔW||/||W||)
FROB_LIGHT = 0.015      # Qwen-like: surgical fine-tuning
FROB_MODERATE = 0.05    # Llama-3.1/Mistral: standard SFT
FROB_HEAVY = 0.12       # Llama-3.2/Gemma-2: heavy rewriting

# Cosine similarity
COS_VERY_HIGH = 0.9999  # Nearly identical weights
COS_HIGH = 0.999        # Standard SFT
COS_MODERATE = 0.99     # Significant changes

# Concentration (top-k energy fraction)
CONC_LOW_RANK = 0.15    # Changes concentrated in few directions
CONC_FULL_RANK = 0.08   # Changes spread across many directions

# Attention vs MLP ratio
ATTN_MLP_BALANCED = 0.95  # Above = balanced, below = MLP-heavy

# LayerNorm change
LN_UNTOUCHED = 0.001    # Below = LayerNorm basically unchanged
LN_MODIFIED = 0.01      # Above = LayerNorm was intentionally changed

# U-shape layer profile
U_SHAPE_THRESHOLD = 0.92  # Below = U-shaped, above = flat/inverted


@dataclass
class Diagnosis:
    """A single diagnostic finding."""
    category: str       # e.g. "magnitude", "structure", "attention"
    severity: str       # "info", "notable", "warning"
    title: str          # Short heading
    detail: str         # 1-2 sentence explanation
    metric_value: float | str = ""


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for a model pair."""
    model_a: str
    model_b: str
    summary: str = ""               # One-paragraph summary
    findings: list[Diagnosis] = field(default_factory=list)
    profile_tag: str = ""           # e.g. "surgical", "standard", "heavy"


def _layer_profile(modules: list[dict]) -> dict:
    """Compute per-layer norms and U-shape ratio."""
    layer_norms: dict[int, float] = {}
    for m in modules:
        match = re.search(r"layers\.(\d+)", m["name"])
        if match:
            li = int(match.group(1))
            layer_norms.setdefault(li, 0.0)
            layer_norms[li] += m["frob_norm"] ** 2
    layer_norms = {k: math.sqrt(v) for k, v in layer_norms.items()}

    if not layer_norms:
        return {"u_shape_ratio": 1.0, "n_layers": 0, "peak_layer": 0, "layer_norms": {}}

    layers = sorted(layer_norms.keys())
    n = len(layers)
    q1 = sum(layer_norms[l] for l in layers[: n // 4]) / max(1, n // 4)
    q4 = sum(layer_norms[l] for l in layers[3 * n // 4 :]) / max(1, n - 3 * n // 4)
    mid = sum(layer_norms[l] for l in layers[n // 4 : 3 * n // 4]) / max(1, n // 2)
    u_ratio = (q1 + q4) / (2 * mid + 1e-12)

    peak_layer = max(layer_norms, key=layer_norms.get)

    return {
        "u_shape_ratio": u_ratio,
        "n_layers": n,
        "peak_layer": peak_layer,
        "layer_norms": layer_norms,
    }


def _module_groups(modules: list[dict]) -> dict:
    """Split modules into functional groups."""
    attn = [m for m in modules if "attn" in m["name"] and m.get("has_svd")]
    mlp = [m for m in modules if "mlp" in m["name"] and m.get("has_svd")]
    lm_head = [m for m in modules if "lm_head" in m["name"] or "embed_tokens" in m["name"]]
    layernorm = [
        m for m in modules
        if any(kw in m["name"].lower() for kw in ("layernorm", "ln", "rmsnorm", "norm"))
        and "attn" not in m["name"] and "mlp" not in m["name"]
    ]
    svd_mods = [m for m in modules if m.get("has_svd")]
    return {
        "attn": attn, "mlp": mlp, "lm_head": lm_head,
        "layernorm": layernorm, "svd": svd_mods,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def diagnose(
    modules: list[dict],
    model_a: str,
    model_b: str,
) -> DiagnosticReport:
    """Generate diagnostic conclusions from weight diff results."""
    report = DiagnosticReport(model_a=model_a, model_b=model_b)
    findings = report.findings
    groups = _module_groups(modules)
    profile = _layer_profile(modules)

    # ── 1. Overall magnitude ────────────────────────────
    mean_frob = _mean([m["frob_norm_relative"] for m in modules])
    mean_cos = _mean([min(1.0, m["cosine_sim"]) for m in modules])
    total_frob = math.sqrt(sum(m["frob_norm"] ** 2 for m in modules))

    if mean_frob < FROB_LIGHT:
        report.profile_tag = "surgical"
        findings.append(Diagnosis(
            "magnitude", "info",
            "Surgical fine-tuning",
            f"Mean relative change is very small ({mean_frob:.4f}). "
            f"This model was fine-tuned conservatively — weights barely moved from base. "
            f"Typical of careful SFT with low learning rate or few steps.",
            mean_frob,
        ))
    elif mean_frob < FROB_MODERATE:
        report.profile_tag = "standard"
        findings.append(Diagnosis(
            "magnitude", "info",
            "Standard fine-tuning",
            f"Mean relative change ({mean_frob:.4f}) is in the typical range for SFT. "
            f"The model was modified enough to learn new behaviors "
            f"while staying close to base model capabilities.",
            mean_frob,
        ))
    elif mean_frob < FROB_HEAVY:
        report.profile_tag = "heavy"
        findings.append(Diagnosis(
            "magnitude", "notable",
            "Heavy fine-tuning",
            f"Mean relative change ({mean_frob:.4f}) is above average. "
            f"Significant weight modifications suggest aggressive SFT, "
            f"multi-stage training (SFT+DPO/RLHF), or large dataset.",
            mean_frob,
        ))
    else:
        report.profile_tag = "extreme"
        findings.append(Diagnosis(
            "magnitude", "warning",
            "Extreme fine-tuning",
            f"Mean relative change ({mean_frob:.4f}) is very high. "
            f"Weights have moved dramatically from base — this could indicate "
            f"continued pre-training, very aggressive SFT, or a substantially different model.",
            mean_frob,
        ))

    # ── 2. Structural analysis (concentration / rank) ───
    svd_mods = groups["svd"]
    if svd_mods:
        mean_conc = _mean([
            m["concentration_top_k"] for m in svd_mods if m.get("concentration_top_k")
        ])
        mean_eff_rank = _mean([
            m["effective_rank"] for m in svd_mods if m.get("effective_rank")
        ])

        if mean_conc > CONC_LOW_RANK:
            findings.append(Diagnosis(
                "structure", "info",
                "Semi-structured changes (LoRA-like directions)",
                f"Top-20 singular values capture {mean_conc:.1%} of change energy. "
                f"Changes are partially concentrated in a low-rank subspace. "
                f"This is consistent with targeted behavioral adjustments "
                f"(e.g., output format, safety guardrails).",
                mean_conc,
            ))
        elif mean_conc > CONC_FULL_RANK:
            findings.append(Diagnosis(
                "structure", "info",
                "Moderate-rank changes",
                f"Top-20 concentration is {mean_conc:.1%} — changes are spread across "
                f"many directions but not uniformly. Effective rank ≈ {mean_eff_rank:.0f}. "
                f"Typical of standard SFT on diverse data.",
                mean_conc,
            ))
        else:
            findings.append(Diagnosis(
                "structure", "notable",
                "Full-rank changes (broad weight rewriting)",
                f"Top-20 concentration is only {mean_conc:.1%} — changes are spread across "
                f"thousands of directions (eff. rank ≈ {mean_eff_rank:.0f}). "
                f"This is NOT LoRA-like — the full parameter space was utilized. "
                f"Suggests extensive training, possibly continued pre-training.",
                mean_conc,
            ))

    # ── 3. Attention vs MLP ─────────────────────────────
    attn_mods = groups["attn"]
    mlp_mods = groups["mlp"]
    if attn_mods and mlp_mods:
        attn_mean = _mean([m["frob_norm_relative"] for m in attn_mods])
        mlp_mean = _mean([m["frob_norm_relative"] for m in mlp_mods])
        ratio = attn_mean / (mlp_mean + 1e-12)

        if ratio > ATTN_MLP_BALANCED:
            findings.append(Diagnosis(
                "attention", "info",
                "Balanced attention/MLP changes",
                f"Attention ({attn_mean:.4f}) and MLP ({mlp_mean:.4f}) changed roughly equally "
                f"(ratio {ratio:.2f}). Fine-tuning affected both knowledge retrieval and "
                f"reasoning/generation pathways uniformly.",
                ratio,
            ))
        else:
            findings.append(Diagnosis(
                "attention", "notable",
                "MLP changed more than attention",
                f"MLP layers ({mlp_mean:.4f}) changed more than attention ({attn_mean:.4f}), "
                f"ratio {ratio:.2f}. This suggests the fine-tuning primarily modified "
                f"the model's knowledge/feed-forward representations rather than "
                f"attention patterns.",
                ratio,
            ))

    # ── 4. Head and embeddings ──────────────────────────
    lm_head = groups["lm_head"]
    if lm_head:
        max_lm = max(m["frob_norm_relative"] for m in lm_head)
        body_mean = _mean([
            m["frob_norm_relative"] for m in modules
            if "lm_head" not in m["name"] and "embed_tokens" not in m["name"]
        ])
        lm_ratio = max_lm / (body_mean + 1e-12)

        if lm_ratio > 2.0:
            findings.append(Diagnosis(
                "head", "notable",
                "Output head is the most changed module",
                f"lm_head/embed change ({max_lm:.4f}) is {lm_ratio:.1f}× the body average "
                f"({body_mean:.4f}). The output vocabulary distribution was significantly "
                f"reshaped — the model's 'voice' was retrained, not just its reasoning.",
                max_lm,
            ))

    # ── 5. LayerNorm analysis ───────────────────────────
    ln_mods = groups["layernorm"]
    if ln_mods:
        ln_mean = _mean([m["frob_norm_relative"] for m in ln_mods])
        if ln_mean < LN_UNTOUCHED:
            findings.append(Diagnosis(
                "layernorm", "info",
                "LayerNorm weights nearly untouched",
                f"LayerNorm mean change is {ln_mean:.6f} — essentially frozen during training. "
                f"This is normal for standard SFT and means the model's internal "
                f"scale calibration was preserved.",
                ln_mean,
            ))
        elif ln_mean > LN_MODIFIED:
            findings.append(Diagnosis(
                "layernorm", "notable",
                "LayerNorm was significantly modified",
                f"LayerNorm mean change is {ln_mean:.4f} — unusually high. "
                f"This suggests aggressive training that shifted internal representations "
                f"enough to require recalibration of normalization layers.",
                ln_mean,
            ))

    # ── 6. Layer profile (U-shape) ──────────────────────
    if profile["n_layers"] > 4:
        u_ratio = profile["u_shape_ratio"]
        if u_ratio < U_SHAPE_THRESHOLD:
            findings.append(Diagnosis(
                "layer_profile", "info",
                "U-shaped layer profile",
                f"Early and late layers changed more than middle layers "
                f"(U-ratio = {u_ratio:.2f}). This is common in SFT: early layers adapt "
                f"input processing (chat format), late layers adapt output generation, "
                f"while middle layers (core reasoning) stay more stable.",
                u_ratio,
            ))
        else:
            findings.append(Diagnosis(
                "layer_profile", "info",
                "Flat or inverted layer profile",
                f"Changes are distributed relatively evenly across layers "
                f"(U-ratio = {u_ratio:.2f}). No strong early/late bias — "
                f"the entire network was modified uniformly.",
                u_ratio,
            ))

    # ── 7. Outlier modules ──────────────────────────────
    by_frob = sorted(modules, key=lambda m: m["frob_norm_relative"], reverse=True)
    if len(by_frob) >= 3:
        top3 = by_frob[:3]
        top3_names = [m["name"].split("model.")[-1] for m in top3]
        top3_vals = [f"{m['frob_norm_relative']:.4f}" for m in top3]
        findings.append(Diagnosis(
            "outliers", "info",
            "Most changed modules",
            f"Top-3 by relative change: "
            + ", ".join(f"{n} ({v})" for n, v in zip(top3_names, top3_vals))
            + ". These modules absorbed the most fine-tuning signal.",
        ))

    # ── Build summary paragraph ─────────────────────────
    profile_desc = {
        "surgical": "surgically fine-tuned with minimal weight changes",
        "standard": "fine-tuned with standard SFT intensity",
        "heavy": "heavily fine-tuned with significant weight modifications",
        "extreme": "extensively modified with extreme weight changes",
    }

    conc_desc = ""
    if svd_mods:
        mean_conc = _mean([
            m["concentration_top_k"] for m in svd_mods if m.get("concentration_top_k")
        ])
        if mean_conc > CONC_LOW_RANK:
            conc_desc = " Changes are partially low-rank (LoRA-like structure)."
        elif mean_conc < CONC_FULL_RANK:
            conc_desc = " Changes are full-rank — the entire parameter space was utilized."

    report.summary = (
        f"{model_b} was {profile_desc.get(report.profile_tag, 'modified')} "
        f"relative to {model_a}. "
        f"Mean relative change: {mean_frob:.4f}, "
        f"cosine similarity: {mean_cos:.5f}, "
        f"total ||ΔW||: {total_frob:.2f}.{conc_desc}"
    )

    return report


def format_diagnosis_text(report: DiagnosticReport) -> str:
    """Format diagnosis as plain text for CLI output."""
    lines = [
        "",
        "─── Diagnosis ───",
        "",
        report.summary,
        "",
    ]
    for f in report.findings:
        icon = {"info": "•", "notable": "▸", "warning": "⚠"}[f.severity]
        lines.append(f"  {icon} {f.title}")
        lines.append(f"    {f.detail}")
        lines.append("")
    return "\n".join(lines)


def format_diagnosis_html(report: DiagnosticReport) -> str:
    """Format diagnosis as HTML fragment for embedding in report (dark theme)."""
    severity_colors = {
        "info": ("rgba(56,189,248,0.1)", "#38bdf8"),
        "notable": ("rgba(245,158,11,0.1)", "#f59e0b"),
        "warning": ("rgba(239,68,68,0.1)", "#ef4444"),
    }

    cards = ""
    for f in report.findings:
        bg, border = severity_colors[f.severity]
        cards += f"""<div style='background:{bg};border-left:4px solid {border};
            padding:12px 16px;margin:8px 0;border-radius:6px'>
            <strong style='color:#f1f5f9'>{f.title}</strong><br>
            <span style='font-size:13px;color:#94a3b8'>{f.detail}</span>
        </div>\n"""

    tag_colors = {
        "surgical": "#10b981",
        "standard": "#3b82f6",
        "heavy": "#f59e0b",
        "extreme": "#ef4444",
    }
    tag_color = tag_colors.get(report.profile_tag, "#6b7280")

    return f"""<div style='margin:24px 0'>
<h2 style='color:#e2e8f0;margin-bottom:16px'>Diagnostic Summary</h2>
<div style='background:#1e293b;border-radius:10px;padding:20px;
    border:1px solid #334155;margin-bottom:16px'>
    <span style='display:inline-block;background:{tag_color};color:white;
        padding:4px 14px;border-radius:12px;font-size:12px;font-weight:700;
        letter-spacing:0.05em;margin-bottom:12px'>{report.profile_tag.upper()}</span>
    <p style='font-size:15px;line-height:1.7;margin:10px 0;color:#cbd5e1'>{report.summary}</p>
</div>
{cards}
</div>"""
