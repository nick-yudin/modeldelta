"""Weight delta analysis: norms, cosine similarity, SVD metrics."""

from __future__ import annotations

import gc
import time

import torch


def effective_rank(sigma: torch.Tensor) -> float:
    """Shannon entropy-based effective rank from singular values."""
    s = sigma / sigma.sum()
    s = s[s > 1e-12]
    return float(torch.exp(-(s * torch.log(s)).sum()))


def cosine_sim_clamped(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity clamped to [-1, 1]."""
    cos = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0)
    ).item()
    return float(max(-1.0, min(1.0, cos)))


def analyze_delta(
    w_a: torch.Tensor,
    w_b: torch.Tensor,
    top_k: int = 20,
    sparsity_threshold: float = 1e-5,
) -> dict:
    """Compute all v0 metrics for a single weight delta.

    Memory-optimized: frees w_a/w_b before SVD so peak RAM ≈ 2x tensor size
    instead of 6x.  Full SVD (all singular values) is still computed.
    """
    shape = w_a.shape
    n_params = int(w_a.numel())

    if w_a.ndim == 1:
        delta = (w_b - w_a).float()
        frob = float(delta.norm())
        return {
            "shape": list(shape),
            "n_params": n_params,
            "frob_norm": frob,
            "frob_norm_relative": frob / (float(w_a.float().norm()) + 1e-12),
            "cosine_sim": cosine_sim_clamped(w_a.float().flatten(), w_b.float().flatten()),
            "sparsity": float((delta.abs() < sparsity_threshold).float().mean()),
            "has_svd": False,
        }

    # ── Phase 1: compute metrics that need both w_a and w_b ───
    w_a_2d = w_a.float().reshape(shape[0], -1)
    w_b_2d = w_b.float().reshape(shape[0], -1)

    norm_a = float(w_a_2d.norm())
    cos = cosine_sim_clamped(w_a_2d.flatten(), w_b_2d.flatten())

    # Compute delta in-place on w_b_2d to avoid a third large allocation
    delta_2d = w_b_2d.sub_(w_a_2d)  # w_b_2d is now delta, no extra copy

    frob = float(delta_2d.norm())
    frob_rel = frob / (norm_a + 1e-12)
    sparsity = float((delta_2d.abs() < sparsity_threshold).float().mean())

    # ── Free w_a, w_b before SVD (biggest RAM saver) ──────────
    del w_a_2d, w_a, w_b  # w_b_2d (=delta_2d) is the only large tensor left

    # ── Phase 2: SVD on delta only ────────────────────────────
    m, n = delta_2d.shape
    if min(m, n) > 8192:
        k = min(top_k * 2, min(m, n))
        Q, _ = torch.linalg.qr(delta_2d @ torch.randn(n, k))
        B = Q.T @ delta_2d
        del Q
        _, sigma, _ = torch.linalg.svd(B, full_matrices=False)
        del B
        sigma = sigma[:top_k]
    else:
        sigma = torch.linalg.svdvals(delta_2d)

    del delta_2d  # free before computing derived metrics

    eff_rank = effective_rank(sigma)
    top_k_actual = min(top_k, len(sigma))
    top_sigmas = sigma[:top_k_actual]

    total_energy = float((sigma**2).sum())
    top_k_energy = float((top_sigmas**2).sum())
    concentration = top_k_energy / (total_energy + 1e-12)

    # Spectral decay: fit log(sigma) ~ -alpha * log(rank)
    log_sigma = torch.log(sigma[sigma > 1e-12])
    log_ranks = torch.log(torch.arange(1, len(log_sigma) + 1).float())
    if len(log_sigma) > 2:
        A = torch.stack([log_ranks, torch.ones_like(log_ranks)], dim=1)
        result = torch.linalg.lstsq(A, log_sigma)
        spectral_alpha = float(-result.solution[0])
    else:
        spectral_alpha = None

    return {
        "shape": list(shape),
        "n_params": n_params,
        "frob_norm": frob,
        "frob_norm_relative": frob_rel,
        "cosine_sim": cos,
        "sparsity": sparsity,
        "has_svd": True,
        "effective_rank": eff_rank,
        "concentration_top_k": concentration,
        "top_k": top_k_actual,
        "top_singular_values": top_sigmas.tolist(),
        "spectral_alpha": spectral_alpha,
        "n_singular_values": int(len(sigma)),
    }


def compare_models(
    tensor_map_a: dict[str, str],
    tensor_map_b: dict[str, str],
    top_k: int = 20,
    sparsity_threshold: float = 1e-5,
    min_params: int = 1000,
    progress_callback=None,
) -> tuple[list[dict], int]:
    """Compare all common tensors between two models.

    Returns (results_list, n_skipped).
    """
    from modeldelta.utils.model_loader import load_tensor

    names_a = set(tensor_map_a.keys())
    names_b = set(tensor_map_b.keys())
    common = sorted(names_a & names_b)

    results = []
    skipped = 0
    t0 = time.time()

    for i, name in enumerate(common):
        w_a = load_tensor(tensor_map_a, name)
        w_b = load_tensor(tensor_map_b, name)

        if w_a.numel() < min_params or w_a.shape != w_b.shape:
            skipped += 1
            del w_a, w_b
            continue

        # analyze_delta deletes its internal refs before SVD,
        # but caller refs also need to go for actual deallocation
        metrics = analyze_delta(w_a, w_b, top_k=top_k, sparsity_threshold=sparsity_threshold)
        del w_a, w_b
        metrics["name"] = name
        results.append(metrics)

        gc.collect()

        if progress_callback and ((i + 1) % 20 == 0 or (i + 1) == len(common)):
            progress_callback(i + 1, len(common), time.time() - t0, name)

    return results, skipped
