"""Landing page generator — unified portal for model-diff.

Generates a single-file HTML page with:
  1. Hero section with tool description
  2. Gallery of pre-computed pairs (clickable → opens full report)
  3. "Compare your own" form (≤3B online, or generate CLI/Colab)
  4. How it works section
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class PairCard:
    """Summary card for a pre-computed pair."""
    model_a: str
    model_b: str
    profile_tag: str
    mean_frob: float
    mean_cos: float
    mean_eff_rank: float
    mean_conc: float
    n_tensors: int
    diagnosis_summary: str
    report_filename: str = ""  # link to full HTML report


def pairs_from_results_dir(results_dir: str) -> list[PairCard]:
    """Load pair cards from a directory of JSON result files."""
    import os, math
    cards = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        modules = data.get("modules", data) if isinstance(data, dict) else data
        if not isinstance(modules, list) or not modules:
            continue

        from model_diff.report.diagnostics import diagnose
        model_a = data.get("model_a", fname.replace(".json", ""))
        model_b = data.get("model_b", fname.replace(".json", ""))
        diag = diagnose(modules, model_a, model_b)

        svd_mods = [m for m in modules if m.get("has_svd")]
        def _mean(vals): return sum(vals) / max(1, len(vals))

        cards.append(PairCard(
            model_a=model_a,
            model_b=model_b,
            profile_tag=diag.profile_tag,
            mean_frob=_mean([m["frob_norm_relative"] for m in modules]),
            mean_cos=_mean([min(1.0, m["cosine_sim"]) for m in modules]),
            mean_eff_rank=_mean([m["effective_rank"] for m in svd_mods if m.get("effective_rank")]),
            mean_conc=_mean([m["concentration_top_k"] for m in svd_mods if m.get("concentration_top_k")]),
            n_tensors=len(modules),
            diagnosis_summary=diag.summary,
            report_filename=fname.replace(".json", ".html"),
        ))
    return cards


def generate_landing_page(
    pairs: list[PairCard] | None = None,
    results_dir: str | None = None,
) -> str:
    """Generate the landing page HTML."""
    if pairs is None and results_dir:
        pairs = pairs_from_results_dir(results_dir)
    if pairs is None:
        pairs = []

    # Sort: extreme first (most interesting)
    order = {"extreme": 0, "heavy": 1, "standard": 2, "surgical": 3}
    pairs.sort(key=lambda p: (order.get(p.profile_tag, 9), -p.mean_frob))

    # Build gallery cards
    gallery_cards = ""
    for i, p in enumerate(pairs):
        tag_colors = {
            "surgical": "#10b981",
            "standard": "#3b82f6",
            "heavy": "#f59e0b",
            "extreme": "#ef4444",
        }
        accent = tag_colors.get(p.profile_tag, "#6b7280")

        short_a = p.model_a.split("/")[-1] if "/" in p.model_a else p.model_a
        short_b = p.model_b.split("/")[-1] if "/" in p.model_b else p.model_b

        report_link = f"reports/{p.report_filename}" if p.report_filename else "#"

        gallery_cards += f"""
        <div class="pair-card" style="border-top: 3px solid {accent};"
             onclick="document.getElementById('detail-{i}').classList.toggle('hidden')">
          <div class="pair-header">
            <span class="tag" style="background:{accent}">{p.profile_tag.upper()}</span>
            <span class="pair-frob" style="color:{accent}">{p.mean_frob:.4f}</span>
          </div>
          <div class="pair-models">
            <span class="model-name">{short_a}</span>
            <span class="arrow">&#x2192;</span>
            <span class="model-name">{short_b}</span>
          </div>
          <div class="pair-metrics">
            <span title="Cosine similarity">cos {p.mean_cos:.5f}</span>
            <span title="Effective rank">rank {p.mean_eff_rank:.0f}</span>
            <span title="Top-20 concentration">conc {p.mean_conc:.3f}</span>
          </div>
          <div id="detail-{i}" class="pair-detail hidden">
            <p>{p.diagnosis_summary}</p>
            <a class="btn-report" href="{report_link}" onclick="event.stopPropagation()">
              Full report &#x2192;
            </a>
          </div>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>model-diff — See what changed inside any model</title>
<style>
  :root {{
    --bg: #0f172a;
    --bg2: #1e293b;
    --bg3: #334155;
    --text: #e2e8f0;
    --text2: #94a3b8;
    --accent: #38bdf8;
    --accent2: #818cf8;
    --green: #10b981;
    --orange: #f59e0b;
    --red: #ef4444;
    --radius: 12px;
  }}

  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }}

  /* ── Hero ── */
  .hero {{
    text-align: center;
    padding: 80px 20px 60px;
    background: linear-gradient(135deg, var(--bg) 0%, #1a1a2e 50%, #16213e 100%);
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(56,189,248,0.08) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(129,140,248,0.06) 0%, transparent 50%);
    animation: pulse 8s ease-in-out infinite alternate;
  }}
  @keyframes pulse {{
    0% {{ transform: scale(1); opacity: 0.5; }}
    100% {{ transform: scale(1.1); opacity: 1; }}
  }}
  .hero h1 {{
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    position: relative;
    margin-bottom: 16px;
  }}
  .hero h1 .highlight {{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }}
  .hero .tagline {{
    font-size: 1.25rem;
    color: var(--text2);
    max-width: 600px;
    margin: 0 auto 40px;
    position: relative;
  }}

  /* ── Action buttons ── */
  .actions {{
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
    position: relative;
    margin-bottom: 20px;
  }}
  .btn {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
  }}
  .btn-primary {{
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #0f172a;
  }}
  .btn-primary:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(56,189,248,0.3); }}
  .btn-secondary {{
    background: var(--bg3);
    color: var(--text);
    border: 1px solid rgba(255,255,255,0.1);
  }}
  .btn-secondary:hover {{ background: #475569; transform: translateY(-2px); }}

  .install-cmd {{
    display: inline-block;
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 8px;
    padding: 12px 20px;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 14px;
    color: var(--accent);
    margin-top: 20px;
    position: relative;
    cursor: pointer;
  }}
  .install-cmd:hover {{ border-color: var(--accent); }}

  /* ── Sections ── */
  .section {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 60px 20px;
  }}
  .section-title {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 8px;
  }}
  .section-sub {{
    color: var(--text2);
    margin-bottom: 32px;
    font-size: 1.05rem;
  }}

  /* ── Gallery ── */
  .gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
    gap: 16px;
  }}
  .pair-card {{
    background: var(--bg2);
    border-radius: var(--radius);
    padding: 20px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .pair-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    background: var(--bg3);
  }}
  .pair-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .tag {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.05em;
  }}
  .pair-frob {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6rem;
    font-weight: 800;
  }}
  .pair-models {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .model-name {{
    font-weight: 700;
    font-size: 15px;
    color: #f1f5f9;
  }}
  .arrow {{
    color: var(--text2);
    font-size: 18px;
  }}
  .pair-metrics {{
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: var(--text2);
    font-family: 'JetBrains Mono', monospace;
  }}
  .pair-detail {{
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--bg3);
    font-size: 13px;
    color: var(--text2);
    line-height: 1.7;
  }}
  .btn-report {{
    display: inline-block;
    margin-top: 12px;
    padding: 6px 16px;
    background: var(--accent);
    color: #0f172a;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    transition: all 0.2s;
  }}
  .btn-report:hover {{ background: #7dd3fc; transform: translateY(-1px); }}
  .hidden {{ display: none; }}

  /* ── Compare form ── */
  .compare-section {{
    background: var(--bg2);
    border-radius: var(--radius);
    padding: 32px;
    margin-top: 20px;
  }}
  .form-row {{
    display: flex;
    gap: 12px;
    margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .form-input {{
    flex: 1;
    min-width: 200px;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid var(--bg3);
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    font-family: inherit;
  }}
  .form-input:focus {{ outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(56,189,248,0.15); }}
  .form-input::placeholder {{ color: var(--text2); }}

  .method-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
    margin-top: 20px;
  }}
  .method-card {{
    background: var(--bg);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    padding: 20px;
    transition: all 0.2s;
  }}
  .method-card:hover {{ border-color: var(--accent); }}
  .method-card h4 {{
    font-size: 15px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .method-card p {{
    font-size: 13px;
    color: var(--text2);
    margin-bottom: 12px;
  }}
  .method-card code {{
    display: block;
    background: var(--bg2);
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    color: var(--accent);
    overflow-x: auto;
  }}
  .method-icon {{
    font-size: 20px;
  }}

  /* ── How it works ── */
  .steps {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 20px;
    counter-reset: step;
  }}
  .step {{
    background: var(--bg2);
    border-radius: var(--radius);
    padding: 24px;
    position: relative;
  }}
  .step::before {{
    counter-increment: step;
    content: counter(step);
    position: absolute;
    top: -12px;
    left: 20px;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 13px;
    color: #0f172a;
  }}
  .step h4 {{ margin-bottom: 8px; font-size: 15px; }}
  .step p {{ font-size: 13px; color: var(--text2); }}

  /* ── Profile legend ── */
  .profiles {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
  }}
  .profile-card {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--bg2);
    border-radius: 8px;
    padding: 16px;
  }}
  .profile-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .profile-info {{ font-size: 13px; }}
  .profile-info strong {{ display: block; font-size: 14px; }}

  /* ── Footer ── */
  footer {{
    text-align: center;
    padding: 40px 20px;
    color: var(--text2);
    font-size: 13px;
    border-top: 1px solid var(--bg3);
  }}
  footer a {{ color: var(--accent); text-decoration: none; }}
  footer a:hover {{ text-decoration: underline; }}

  /* ── Responsive ── */
  @media (max-width: 640px) {{
    .gallery {{ grid-template-columns: 1fr; }}
    .pair-metrics {{ flex-wrap: wrap; gap: 8px; }}
  }}
</style>
</head><body>

<!-- ═══ HERO ═══ -->
<div class="hero">
  <h1>
    <span class="highlight">model-diff</span>
  </h1>
  <p class="tagline">
    See what changed inside any model checkpoint.<br>
    Weight deltas, SVD structure, and a diagnostic verdict.
  </p>
  <div class="actions">
    <a class="btn btn-primary" href="#compare">Compare models</a>
    <a class="btn btn-secondary" href="#gallery">Browse results</a>
  </div>
  <div class="install-cmd" onclick="navigator.clipboard.writeText('pip install model-diff')" title="Click to copy">
    $ pip install model-diff
  </div>
</div>

<!-- ═══ GALLERY ═══ -->
<div class="section" id="gallery">
  <h2 class="section-title">Pre-computed comparisons</h2>
  <p class="section-sub">Click any card to see the diagnostic summary. {len(pairs)} base&#x2192;instruct pairs analyzed.</p>

  <div class="gallery">
    {gallery_cards}
  </div>
</div>

<!-- ═══ PROFILES LEGEND ═══ -->
<div class="section">
  <h2 class="section-title">Diagnostic profiles</h2>
  <p class="section-sub">model-diff classifies fine-tuning intensity into four categories.</p>
  <div class="profiles">
    <div class="profile-card">
      <div class="profile-dot" style="background: var(--green)"></div>
      <div class="profile-info">
        <strong>SURGICAL</strong>
        <span style="color:var(--text2)">&Delta;W/W &lt; 1.5% &mdash; Minimal, targeted changes. Qwen-style.</span>
      </div>
    </div>
    <div class="profile-card">
      <div class="profile-dot" style="background: var(--accent)"></div>
      <div class="profile-info">
        <strong>STANDARD</strong>
        <span style="color:var(--text2)">1.5&ndash;5% &mdash; Typical SFT. Llama-3.1, Mistral.</span>
      </div>
    </div>
    <div class="profile-card">
      <div class="profile-dot" style="background: var(--orange)"></div>
      <div class="profile-info">
        <strong>HEAVY</strong>
        <span style="color:var(--text2)">5&ndash;12% &mdash; Aggressive training, LayerNorm may shift.</span>
      </div>
    </div>
    <div class="profile-card">
      <div class="profile-dot" style="background: var(--red)"></div>
      <div class="profile-info">
        <strong>EXTREME</strong>
        <span style="color:var(--text2)">&gt;12% &mdash; Full rewriting. Gemma-2-9B territory.</span>
      </div>
    </div>
  </div>
</div>

<!-- ═══ COMPARE ═══ -->
<div class="section" id="compare">
  <h2 class="section-title">Compare your own models</h2>
  <p class="section-sub">Choose the method that fits your models.</p>

  <div class="compare-section">
    <div class="form-row">
      <input class="form-input" id="model-a" placeholder="Base model (e.g. Qwen/Qwen2.5-3B)" />
      <input class="form-input" id="model-b" placeholder="Fine-tuned model (e.g. Qwen/Qwen2.5-3B-Instruct)" />
    </div>
    <div class="method-cards">
      <div class="method-card">
        <h4><span class="method-icon">&#x1F4BB;</span> Local CLI</h4>
        <p>Any model size. Needs ~6.6 GB RAM for 7B. CPU-only, no GPU required.</p>
        <code id="cli-cmd">model-diff Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct -o report.html</code>
      </div>
      <div class="method-card">
        <h4><span class="method-icon">&#x2601;</span> Google Colab</h4>
        <p>For models &gt;3B. Free CPU runtime. ~18 min per 7B pair.</p>
        <code>Coming soon &mdash; notebook generator</code>
      </div>
      <div class="method-card">
        <h4><span class="method-icon">&#x26A1;</span> Online (&#x2264;3B)</h4>
        <p>Small models computed in-browser. No install needed.</p>
        <code>Coming soon &mdash; HF Space</code>
      </div>
    </div>
  </div>
</div>

<!-- ═══ HOW IT WORKS ═══ -->
<div class="section">
  <h2 class="section-title">How it works</h2>
  <p class="section-sub">No GPU. No full model loading. Just math on weight deltas.</p>
  <div class="steps">
    <div class="step">
      <h4>Stream tensors</h4>
      <p>Downloads only safetensors files. Loads one tensor at a time &mdash; never both full models in RAM.</p>
    </div>
    <div class="step">
      <h4>Compute deltas</h4>
      <p>For each weight matrix: Frobenius norm, cosine similarity, sparsity. Memory-optimized in-place subtraction.</p>
    </div>
    <div class="step">
      <h4>SVD analysis</h4>
      <p>Full singular value decomposition of each delta. Effective rank, top-k concentration, spectral decay.</p>
    </div>
    <div class="step">
      <h4>Diagnose</h4>
      <p>Automatic classification: surgical, standard, heavy, or extreme. Human-readable conclusions, not just numbers.</p>
    </div>
  </div>
</div>

<footer>
  <p>model-diff v{__import__('model_diff').__version__} &middot; MIT License &middot;
  <a href="https://github.com/nick-yudin/model-diff">GitHub</a> &middot;
  <a href="https://pypi.org/project/model-diff/">PyPI</a></p>
</footer>

<script>
// Update CLI command from form inputs
const modelA = document.getElementById('model-a');
const modelB = document.getElementById('model-b');
const cliCmd = document.getElementById('cli-cmd');
function updateCmd() {{
  const a = modelA.value || 'Qwen/Qwen2.5-3B';
  const b = modelB.value || 'Qwen/Qwen2.5-3B-Instruct';
  cliCmd.textContent = `model-diff ${{a}} ${{b}} -o report.html`;
}}
modelA.addEventListener('input', updateCmd);
modelB.addEventListener('input', updateCmd);
</script>

</body></html>"""
