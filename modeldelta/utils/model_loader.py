"""Memory-efficient model loading via safetensors streaming or torch.load."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


@contextmanager
def _suppress_hf_progress():
    """Temporarily disable huggingface_hub progress bars."""
    try:
        from huggingface_hub.utils import disable_progress_bars, enable_progress_bars
        disable_progress_bars()
        yield
        enable_progress_bars()
    except ImportError:
        yield


def download_safetensors(model_id: str, token: str | None = None, quiet: bool = False) -> str:
    """Download model weights (safetensors or pytorch_model.bin). Returns local path."""
    kwargs = dict(
        repo_id=model_id,
        allow_patterns=[
            "*.safetensors", "*.safetensors.index.json",
            "pytorch_model*.bin", "pytorch_model.bin.index.json",
            "config.json",
        ],
        token=token or os.environ.get("HF_TOKEN"),
    )
    if quiet:
        with _suppress_hf_progress():
            return snapshot_download(**kwargs)
    return snapshot_download(**kwargs)


def get_tensor_map(model_path: str) -> dict[str, str]:
    """Build dict: tensor_name -> shard_file_path.

    Supports safetensors (preferred) and pytorch_model.bin formats.
    """
    # ── safetensors sharded ───────────────────────────────────
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        return {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}

    # ── safetensors single file ───────────────────────────────
    sf_file = os.path.join(model_path, "model.safetensors")
    if os.path.exists(sf_file):
        with safe_open(sf_file, framework="pt") as f:
            return {name: sf_file for name in f.keys()}

    # ── pytorch_model.bin sharded ─────────────────────────────
    bin_index = os.path.join(model_path, "pytorch_model.bin.index.json")
    if os.path.exists(bin_index):
        with open(bin_index) as f:
            index = json.load(f)
        return {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}

    # ── pytorch_model.bin single file ─────────────────────────
    bin_file = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(bin_file):
        state = torch.load(bin_file, map_location="cpu", weights_only=True)
        # Return sentinel: shard path is bin_file, names embedded in map
        return {name: bin_file for name in state.keys()}

    raise FileNotFoundError(
        f"No weights found in {model_path} "
        "(expected model.safetensors, model.safetensors.index.json, "
        "pytorch_model.bin, or pytorch_model.bin.index.json)"
    )


# Cache for loaded .bin shards to avoid reloading the same shard per tensor
_bin_shard_cache: dict[str, dict[str, torch.Tensor]] = {}


def load_tensor(tensor_map: dict[str, str], name: str) -> torch.Tensor:
    """Load a single tensor from safetensors or .bin shard."""
    shard_path = tensor_map[name]

    if shard_path.endswith(".bin"):
        # Load shard once, cache it; evict previous shard if different
        if shard_path not in _bin_shard_cache:
            _bin_shard_cache.clear()  # keep only one shard in memory at a time
            _bin_shard_cache[shard_path] = torch.load(
                shard_path, map_location="cpu", weights_only=True
            )
        return _bin_shard_cache[shard_path][name]

    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def clear_bin_cache() -> None:
    """Free cached .bin shard from memory."""
    _bin_shard_cache.clear()


def resolve_model(model_id_or_path: str, token: str | None = None, quiet: bool = False) -> str:
    """Resolve a model ID or local path to a local directory."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return download_safetensors(model_id_or_path, token=token, quiet=quiet)
