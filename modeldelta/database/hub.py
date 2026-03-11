"""Push/pull results to HuggingFace Dataset repo.

Repo: NikolayYudin/modeldelta-results
Structure:
  index.json          — gallery index (list of pair summaries)
  pairs/{pair_id}.json — full results per pair
"""

from __future__ import annotations

import json
from dataclasses import asdict

REPO_ID = "NikolayYudin/modeldelta-results"


def push_results(
    modules: list[dict],
    model_a: str,
    model_b: str,
    n_skipped: int = 0,
    meta_a=None,
    meta_b=None,
    token: str | None = None,
    computed_by: str = "cli",
    top_k: int = 20,
) -> str:
    """Push comparison results to HF Dataset repo.

    Returns the pair_id of the uploaded result.
    """
    from huggingface_hub import HfApi, hf_hub_download
    from modeldelta.database.schema import (
        make_pair_id, results_to_pair_row,
    )
    from modeldelta.report.diagnostics import diagnose

    api = HfApi(token=token)
    pair_id = make_pair_id(model_a, model_b)

    # Ensure repo exists
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)

    # Diagnostics
    diag = diagnose(modules, model_a, model_b)

    # ── Build pair summary ──
    pair_row = results_to_pair_row(
        modules, model_a, model_b,
        diagnosis_summary=diag.summary,
        profile_tag=diag.profile_tag,
        computed_by=computed_by,
        top_k=top_k,
    )
    pair_row["n_skipped"] = n_skipped

    # Add model metadata if available
    if meta_a is not None:
        pair_row["meta_a"] = meta_a.to_dict()
    if meta_b is not None:
        pair_row["meta_b"] = meta_b.to_dict()

    # ── Build full result (same as JSON report) ──
    full_result = {
        "model_a": model_a,
        "model_b": model_b,
        "n_tensors": len(modules),
        "n_skipped": n_skipped,
        "diagnostics": {
            "summary": diag.summary,
            "profile_tag": diag.profile_tag,
            "findings": [asdict(f) for f in diag.findings],
        },
        "modules": modules,
    }
    if meta_a is not None or meta_b is not None:
        full_result["model_info"] = {}
        if meta_a is not None:
            full_result["model_info"]["model_a"] = meta_a.to_dict()
        if meta_b is not None:
            full_result["model_info"]["model_b"] = meta_b.to_dict()

    # ── Upload full result ──
    full_bytes = json.dumps(full_result, indent=2).encode()
    api.upload_file(
        path_or_fileobj=full_bytes,
        path_in_repo=f"pairs/{pair_id}.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Add {model_a} vs {model_b}",
    )

    # ── Update index.json ──
    # Download current index (or start fresh)
    try:
        index_path = hf_hub_download(
            REPO_ID, "index.json", repo_type="dataset", token=token,
        )
        with open(index_path) as f:
            index = json.load(f)
    except Exception:
        index = []

    # Remove existing entry for this pair (if re-running)
    index = [e for e in index if e.get("pair_id") != pair_id]

    # Build gallery entry (lightweight — no modules)
    gallery_entry = {
        "pair_id": pair_id,
        "model_a": model_a,
        "model_b": model_b,
        "profile_tag": diag.profile_tag,
        "mean_frob_relative": pair_row["mean_frob_relative"],
        "mean_cosine_sim": pair_row["mean_cosine_sim"],
        "mean_effective_rank": pair_row["mean_effective_rank"],
        "mean_concentration": pair_row["mean_concentration"],
        "n_tensors": len(modules),
        "diagnosis_summary": diag.summary,
        "computed_at": pair_row["computed_at"],
        "modeldelta_version": pair_row["modeldelta_version"],
    }
    index.append(gallery_entry)

    # Sort: extreme first
    order = {"extreme": 0, "heavy": 1, "standard": 2, "surgical": 3}
    index.sort(key=lambda e: (order.get(e.get("profile_tag", ""), 9),
                               -e.get("mean_frob_relative", 0)))

    index_bytes = json.dumps(index, indent=2).encode()
    api.upload_file(
        path_or_fileobj=index_bytes,
        path_in_repo="index.json",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"Update index: {model_a} vs {model_b}",
    )

    return pair_id


def fetch_gallery_index(token: str | None = None) -> list[dict]:
    """Download the gallery index from HF Dataset repo."""
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            REPO_ID, "index.json", repo_type="dataset", token=token,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []
