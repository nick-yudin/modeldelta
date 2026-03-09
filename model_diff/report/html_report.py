"""Generate static HTML report with embedded plots."""

from __future__ import annotations

import base64
import math
import re
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_layer_info(name: str):
    m = re.search(r"layers\.(\d+)\.(.+)", name)
    if m:
        return int(m.group(1)), m.group(2)
    return None, name


def _short_name(mod_type: str) -> str:
    return (
        mod_type
        .replace("self_attn.", "attn.")
        .replace("_proj", "")
        .replace("mlp.", "mlp.")
        .replace("input_layernorm", "ln_in")
        .replace("post_attention_layernorm", "ln_post")
    )


def _fig_to_base64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


_DARK_BG = "#1e293b"
_DARK_TEXT = "#e2e8f0"


def _apply_dark_style():
    """Set matplotlib style for dark theme plots."""
    plt.rcParams.update({
        "figure.facecolor": _DARK_BG,
        "axes.facecolor": "#0f172a",
        "axes.edgecolor": "#475569",
        "axes.labelcolor": _DARK_TEXT,
        "text.color": _DARK_TEXT,
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "figure.edgecolor": _DARK_BG,
    })


def _build_matrices(modules: list[dict]):
    """Build layer × module_type matrices for heatmap plotting."""
    layer_modules = {}
    module_types = set()
    max_layer = 0

    for m in modules:
        layer_idx, mod_type = _parse_layer_info(m["name"])
        if layer_idx is not None:
            mod_type = mod_type.replace(".weight", "")
            layer_modules[(layer_idx, mod_type)] = m
            module_types.add(mod_type)
            max_layer = max(max_layer, layer_idx)

    module_types = sorted(module_types)
    n_layers = max_layer + 1

    frob = np.full((n_layers, len(module_types)), np.nan)
    cos = np.full((n_layers, len(module_types)), np.nan)
    rank = np.full((n_layers, len(module_types)), np.nan)
    conc = np.full((n_layers, len(module_types)), np.nan)

    for (li, mt), m in layer_modules.items():
        j = module_types.index(mt)
        frob[li, j] = m["frob_norm_relative"]
        cos[li, j] = min(1.0, m["cosine_sim"])
        if m.get("has_svd") and m.get("effective_rank"):
            rank[li, j] = m["effective_rank"]
        if m.get("has_svd") and m.get("concentration_top_k"):
            conc[li, j] = m["concentration_top_k"]

    return frob, cos, rank, conc, module_types, n_layers, layer_modules


def generate_html(
    modules: list[dict],
    model_a: str,
    model_b: str,
    top_k: int = 20,
    sparsity_threshold: float = 1e-5,
    include_diagnostics: bool = True,
    back_link: str = "",
) -> str:
    """Generate complete HTML report as string."""
    _apply_dark_style()
    by_frob = sorted(modules, key=lambda m: m["frob_norm_relative"], reverse=True)
    frob_matrix, cos_matrix, rank_matrix, conc_matrix, module_types, n_layers, layer_modules = _build_matrices(modules)
    short_labels = [_short_name(mt) for mt in module_types]

    # 1. Heatmaps
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 18))
    for ax, matrix, title, cmap, extra_kw in [
        (axes1[0, 0], frob_matrix, "||ΔW||/||W||", "Reds", {}),
        (axes1[0, 1], cos_matrix, "Cosine similarity", "RdYlGn",
         {"vmin": float(np.nanmin(cos_matrix)), "vmax": 1.0}),
        (axes1[1, 0], rank_matrix, "Effective rank of ΔW", "viridis", {}),
        (axes1[1, 1], conc_matrix, f"Top-{top_k} concentration", "Oranges", {}),
    ]:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", **extra_kw)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Module type", fontsize=9)
        ax.set_ylabel("Layer", fontsize=9)
        ax.set_xticks(range(len(module_types)))
        ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
        fig1.colorbar(im, ax=ax, shrink=0.8)
    fig1.suptitle(f"model-diff: {model_a} → {model_b}", fontsize=13, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    heatmap_b64 = _fig_to_base64(fig1)
    plt.close(fig1)

    # 2. SVD spectra top-9
    svd_modules = [m for m in modules if m.get("has_svd") and m.get("top_singular_values")]
    top_changed = sorted(svd_modules, key=lambda m: m["frob_norm_relative"], reverse=True)[:9]
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    for ax, m in zip(axes2.flat, top_changed):
        sigmas = np.array(m["top_singular_values"])
        ax.semilogy(range(1, len(sigmas) + 1), sigmas, color="#38bdf8", marker=".", markersize=4)
        ax.set_title(m["name"].split("model.")[-1], fontsize=9)
        ax.set_xlabel("rank")
        ax.set_ylabel("σ")
        info = f"eff_rank={m['effective_rank']:.1f}\nconc={m['concentration_top_k']:.3f}"
        if m.get("spectral_alpha"):
            info += f"\nα={m['spectral_alpha']:.2f}"
        ax.text(0.95, 0.95, info, transform=ax.transAxes, fontsize=7,
                va="top", ha="right", color="#e2e8f0",
                bbox=dict(boxstyle="round", facecolor="#334155", alpha=0.8))
    for ax in axes2.flat[len(top_changed):]:
        ax.set_visible(False)
    fig2.suptitle("Top-9 changed modules: singular value spectra", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    svd_b64 = _fig_to_base64(fig2)
    plt.close(fig2)

    # 3. Per-layer bar chart
    layer_norms = np.zeros(n_layers)
    for (li, mt), m in layer_modules.items():
        layer_norms[li] += m["frob_norm"] ** 2
    layer_norms = np.sqrt(layer_norms)
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.bar(range(n_layers), layer_norms, color="#38bdf8")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("||ΔW|| (Frobenius)")
    ax3.set_title("Per-layer total weight delta magnitude")
    plt.tight_layout()
    layer_bar_b64 = _fig_to_base64(fig3)
    plt.close(fig3)

    # Diagnostics
    diag_html = ""
    if include_diagnostics:
        from model_diff.report.diagnostics import diagnose, format_diagnosis_html
        diag_report = diagnose(modules, model_a, model_b)
        diag_html = format_diagnosis_html(diag_report)

    # Summary stats
    svd_mods = [m for m in modules if m.get("has_svd")]
    mean_eff_rank = np.mean([m["effective_rank"] for m in svd_mods if m.get("effective_rank")])
    mean_conc = np.mean([m["concentration_top_k"] for m in svd_mods if m.get("concentration_top_k")])
    mean_cos = np.mean([min(1.0, m["cosine_sim"]) for m in modules])
    mean_sparsity = np.mean([m["sparsity"] for m in modules])
    total_frob = math.sqrt(sum(m["frob_norm"] ** 2 for m in modules))

    # Table rows
    table_rows = ""
    for m in by_frob:
        eff_r = f"{m['effective_rank']:.1f}" if m.get("has_svd") and m.get("effective_rank") else "—"
        conc = f"{m['concentration_top_k']:.3f}" if m.get("has_svd") and m.get("concentration_top_k") else "—"
        alpha = f"{m['spectral_alpha']:.2f}" if m.get("spectral_alpha") else "—"
        table_rows += f"""<tr>
            <td style='font-family:monospace;font-size:12px'>{m['name']}</td>
            <td>{m['n_params']:,}</td>
            <td>{m['frob_norm_relative']:.4f}</td>
            <td>{min(1.0, m['cosine_sim']):.5f}</td>
            <td>{eff_r}</td>
            <td>{conc}</td>
            <td>{m['sparsity']:.3f}</td>
            <td>{alpha}</td>
        </tr>\n"""

    back_btn = f'<a href="{back_link}" class="back-link">&#x2190; Back to gallery</a>' if back_link else ""

    return f"""<!DOCTYPE html>
<html><head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>model-diff: {model_a} → {model_b}</title>
<style>
  :root {{
    --bg: #0f172a; --bg2: #1e293b; --bg3: #334155;
    --text: #e2e8f0; --text2: #94a3b8;
    --accent: #38bdf8; --accent2: #818cf8;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 20px 24px;
    background: var(--bg); color: var(--text); line-height: 1.6;
  }}
  .back-link {{
    display: inline-block; margin-bottom: 20px; padding: 6px 14px;
    background: var(--bg2); border: 1px solid var(--bg3); border-radius: 6px;
    color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 600;
    transition: all 0.2s;
  }}
  .back-link:hover {{ background: var(--bg3); transform: translateY(-1px); }}
  h1 {{
    font-size: 1.75rem; font-weight: 800; color: var(--text);
    border-bottom: 2px solid var(--accent); padding-bottom: 12px; margin-bottom: 8px;
  }}
  .subtitle {{ color: var(--text2); font-size: 1.1rem; margin-bottom: 24px; }}
  h2 {{
    color: var(--text); margin-top: 40px; margin-bottom: 16px;
    font-size: 1.25rem; font-weight: 700;
  }}
  .summary {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin: 20px 0;
  }}
  .card {{
    background: var(--bg2); border-radius: 10px; padding: 16px;
    border: 1px solid var(--bg3); transition: border-color 0.2s;
  }}
  .card:hover {{ border-color: var(--accent); }}
  .card .value {{ font-size: 1.5rem; font-weight: 800; color: var(--accent); }}
  .card .label {{ font-size: 12px; color: var(--text2); margin-top: 4px; }}
  img {{
    max-width: 100%; border-radius: 10px; margin: 12px 0;
    border: 1px solid var(--bg3);
  }}
  table {{
    border-collapse: collapse; width: 100%; border-radius: 10px;
    overflow: hidden; border: 1px solid var(--bg3);
  }}
  th {{
    background: var(--bg3); color: var(--text); padding: 10px 12px;
    text-align: left; font-size: 12px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.05em;
  }}
  td {{
    padding: 8px 12px; border-bottom: 1px solid var(--bg3);
    font-size: 13px; font-family: 'JetBrains Mono', monospace;
    color: var(--text2);
  }}
  tr:hover {{ background: rgba(56,189,248,0.05); }}
  tr:hover td {{ color: var(--text); }}
  .meta {{
    color: var(--text2); font-size: 12px; margin-top: 40px;
    padding-top: 20px; border-top: 1px solid var(--bg3);
  }}
</style>
</head><body>
{back_btn}
<h1>model-diff report</h1>
<p class='subtitle'><strong style="color:var(--text)">{model_a}</strong>
  <span style="color:var(--accent)">&#x2192;</span>
  <strong style="color:var(--text)">{model_b}</strong></p>
<div class='summary'>
  <div class='card'><div class='value'>{len(modules)}</div><div class='label'>Tensors analyzed</div></div>
  <div class='card'><div class='value'>{total_frob:.2f}</div><div class='label'>Total ||&#x394;W||</div></div>
  <div class='card'><div class='value'>{mean_cos:.5f}</div><div class='label'>Mean cosine sim</div></div>
  <div class='card'><div class='value'>{mean_eff_rank:.1f}</div><div class='label'>Mean eff. rank</div></div>
  <div class='card'><div class='value'>{mean_conc:.3f}</div><div class='label'>Mean top-{top_k} conc.</div></div>
  <div class='card'><div class='value'>{mean_sparsity:.3f}</div><div class='label'>Mean sparsity</div></div>
</div>
{diag_html}
<h2>Per-layer delta magnitude</h2>
<img src='data:image/png;base64,{layer_bar_b64}' alt='Layer norms'>
<h2>Layer x Module heatmaps</h2>
<img src='data:image/png;base64,{heatmap_b64}' alt='Heatmaps'>
<h2>SVD spectra (top-9 changed)</h2>
<img src='data:image/png;base64,{svd_b64}' alt='SVD spectra'>
<h2>Full module table</h2>
<table>
<thead><tr>
  <th>Module</th><th>Params</th><th>||&#x394;W||/||W||</th><th>cos_sim</th><th>eff_rank</th><th>conc</th><th>sparsity</th><th>&#x03B1;</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
<div class='meta'>Generated by model-diff v{__import__('model_diff').__version__}</div>
</body></html>"""
