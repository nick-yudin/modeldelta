"""Fetch and validate HuggingFace model metadata.

Provides model cards for reports and pre-flight validation before
downloading tens of GB of safetensors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict


@dataclass
class ModelMeta:
    """Structured metadata for a HuggingFace model."""
    model_id: str
    exists: bool = True
    base_model: str | None = None
    pipeline_tag: str | None = None
    license: str | None = None
    library: str | None = None
    tags: list[str] | None = None
    downloads: int = 0
    likes: int = 0
    n_safetensors: int = 0
    safetensors_gb: float = 0.0
    n_bin: int = 0
    gated: bool = False
    error: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop None/empty for cleaner JSON
        return {k: v for k, v in d.items() if v is not None and v != 0 and v != []}

    def one_liner(self) -> str:
        """Human-readable one-line summary."""
        parts = []
        if self.pipeline_tag:
            parts.append(self.pipeline_tag)
        if self.license:
            parts.append(self.license)
        if self.base_model:
            parts.append(f"from {self.base_model}")
        if self.downloads:
            parts.append(f"{_fmt_count(self.downloads)} downloads")
        if self.tags:
            # Pick interesting tags (skip generic ones)
            skip = {"safetensors", "transformers", "license:" + (self.license or ""),
                    "conversational", "endpoints_compatible", "region:us",
                    self.pipeline_tag or "", self.library or ""}
            interesting = [t for t in self.tags if t not in skip
                           and not t.startswith("license:")
                           and not t.startswith("base_model:")
                           and not t.startswith("arxiv:")][:4]
            if interesting:
                parts.append(", ".join(interesting))
        return " | ".join(parts) if parts else self.model_id


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _normalize_license(val) -> str | None:
    """HF API sometimes returns license as a list instead of str."""
    if val is None:
        return None
    if isinstance(val, list):
        return ", ".join(str(v) for v in val) if val else None
    return str(val)


def fetch_model_meta(model_id: str, token: str | None = None) -> ModelMeta:
    """Fetch metadata from HuggingFace Hub. Fast — no model download."""
    if os.path.isdir(model_id):
        return ModelMeta(model_id=model_id, exists=True)

    try:
        from huggingface_hub import model_info
        info = model_info(model_id, token=token)
    except Exception as e:
        return ModelMeta(
            model_id=model_id,
            exists=False,
            error=str(e),
        )

    card = info.card_data
    safetensors = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    bin_files = [s for s in info.siblings
                 if s.rfilename.endswith(".bin") and "pytorch_model" in s.rfilename]
    total_gb = sum(s.size for s in safetensors if s.size) / 1e9

    return ModelMeta(
        model_id=model_id,
        exists=True,
        base_model=_normalize_license(getattr(card, "base_model", None)) if card else None,
        pipeline_tag=info.pipeline_tag,
        license=_normalize_license(getattr(card, "license", None)) if card else None,
        library=info.library_name,
        tags=info.tags[:15] if info.tags else None,
        downloads=info.downloads or 0,
        likes=info.likes or 0,
        n_safetensors=len(safetensors),
        safetensors_gb=round(total_gb, 2),
        n_bin=len(bin_files),
        gated=bool(info.gated),
    )


def validate_pair(
    model_a: str,
    model_b: str,
    token: str | None = None,
) -> tuple[ModelMeta, ModelMeta, list[str]]:
    """Validate both models and return metadata + warnings.

    Returns:
        (meta_a, meta_b, warnings)
    """
    meta_a = fetch_model_meta(model_a, token=token)
    meta_b = fetch_model_meta(model_b, token=token)
    warnings = []

    if not meta_a.exists:
        warnings.append(f"Model A not found: {model_a}")
    if not meta_b.exists:
        warnings.append(f"Model B not found: {model_b}")

    if meta_a.exists and meta_b.exists:
        # Check weight file availability (safetensors preferred, .bin also supported)
        def _has_weights(meta: ModelMeta) -> bool:
            return meta.n_safetensors > 0 or meta.n_bin > 0

        if not _has_weights(meta_a) and not os.path.isdir(model_a):
            warnings.append(f"{model_a}: no weight files found (safetensors or pytorch_model.bin)")
        if not _has_weights(meta_b) and not os.path.isdir(model_b):
            warnings.append(f"{model_b}: no weight files found (safetensors or pytorch_model.bin)")

        # Check architecture compatibility (shard count, same format only)
        same_format = (meta_a.n_safetensors > 0) == (meta_b.n_safetensors > 0)
        shards_a = meta_a.n_safetensors or meta_a.n_bin
        shards_b = meta_b.n_safetensors or meta_b.n_bin
        if same_format and shards_a > 0 and shards_b > 0 and abs(shards_a - shards_b) > 2:
            warnings.append(
                f"Shard count mismatch: {model_a} has {shards_a}, "
                f"{model_b} has {shards_b} — models may be incompatible"
            )

        # Check base_model consistency
        if meta_b.base_model and meta_b.base_model != model_a:
            if meta_a.base_model is None:
                pass  # model_a might be the base, just no self-reference
            else:
                warnings.append(
                    f"{model_b} declares base_model={meta_b.base_model}, "
                    f"but you're comparing against {model_a}"
                )

    return meta_a, meta_b, warnings
