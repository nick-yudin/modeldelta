"""CLI text report (quick summary table)."""

from __future__ import annotations

import math


def generate_text(
    modules: list[dict],
    model_a: str,
    model_b: str,
    n_skipped: int = 0,
    top_n: int = 20,
    include_diagnostics: bool = True,
) -> str:
    """Generate text summary for terminal output."""
    by_frob = sorted(modules, key=lambda m: m["frob_norm_relative"], reverse=True)
    total_frob = math.sqrt(sum(m["frob_norm"] ** 2 for m in modules))
    svd_mods = [m for m in modules if m.get("has_svd")]
    mean_cos = sum(min(1.0, m["cosine_sim"]) for m in modules) / len(modules)
    mean_eff_rank = (
        sum(m["effective_rank"] for m in svd_mods if m.get("effective_rank"))
        / max(1, len([m for m in svd_mods if m.get("effective_rank")]))
    )

    lines = [
        f"model-diff: {model_a} → {model_b}",
        f"Tensors: {len(modules)} analyzed, {n_skipped} skipped",
        f"Total ||ΔW||: {total_frob:.2f}  |  Mean cos_sim: {mean_cos:.5f}  |  Mean eff_rank: {mean_eff_rank:.0f}",
        "",
        f"{'Module':<55} {'ΔW/W':>8} {'cos_sim':>8} {'eff_rank':>9} {'conc':>7} {'spars':>6}",
        "─" * 95,
    ]

    for m in by_frob[:top_n]:
        eff_r = f"{m['effective_rank']:.0f}" if m.get("has_svd") and m.get("effective_rank") else "—"
        conc = f"{m['concentration_top_k']:.3f}" if m.get("has_svd") and m.get("concentration_top_k") else "—"
        lines.append(
            f"{m['name']:<55} {m['frob_norm_relative']:>8.4f} "
            f"{min(1.0, m['cosine_sim']):>8.5f} {eff_r:>9} {conc:>7} {m['sparsity']:>6.3f}"
        )

    if len(by_frob) > top_n:
        lines.append(f"  ... and {len(by_frob) - top_n} more modules")

    if include_diagnostics:
        from model_diff.report.diagnostics import diagnose, format_diagnosis_text
        diag_report = diagnose(modules, model_a, model_b)
        lines.append(format_diagnosis_text(diag_report))

    return "\n".join(lines)
