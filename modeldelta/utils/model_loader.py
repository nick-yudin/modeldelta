"""Memory-efficient model loading via safetensors streaming."""

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
    """Download only safetensors files + index. Returns local path."""
    kwargs = dict(
        repo_id=model_id,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        token=token or os.environ.get("HF_TOKEN"),
    )
    if quiet:
        with _suppress_hf_progress():
            return snapshot_download(**kwargs)
    return snapshot_download(**kwargs)


def get_tensor_map(model_path: str) -> dict[str, str]:
    """Build dict: tensor_name -> shard_file_path."""
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        return {k: os.path.join(model_path, v) for k, v in index["weight_map"].items()}

    sf_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(sf_file):
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    with safe_open(sf_file, framework="pt") as f:
        return {name: sf_file for name in f.keys()}


def load_tensor(tensor_map: dict[str, str], name: str) -> torch.Tensor:
    """Load a single tensor from safetensors shard."""
    shard_path = tensor_map[name]
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def resolve_model(model_id_or_path: str, token: str | None = None, quiet: bool = False) -> str:
    """Resolve a model ID or local path to a local directory."""
    if os.path.isdir(model_id_or_path):
        return model_id_or_path
    return download_safetensors(model_id_or_path, token=token, quiet=quiet)
