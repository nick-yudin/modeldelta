"""JSON report generation."""

from __future__ import annotations

import json
from dataclasses import asdict


def generate_json(
    modules: list[dict],
    model_a: str,
    model_b: str,
    n_skipped: int = 0,
    include_diagnostics: bool = True,
) -> str:
    """Generate JSON report as string."""
    data = {
        "model_a": model_a,
        "model_b": model_b,
        "n_tensors": len(modules),
        "n_skipped": n_skipped,
        "modules": modules,
    }

    if include_diagnostics:
        from model_diff.report.diagnostics import diagnose
        diag = diagnose(modules, model_a, model_b)
        data["diagnostics"] = {
            "summary": diag.summary,
            "profile_tag": diag.profile_tag,
            "findings": [asdict(f) for f in diag.findings],
        }

    return json.dumps(data, indent=2)
