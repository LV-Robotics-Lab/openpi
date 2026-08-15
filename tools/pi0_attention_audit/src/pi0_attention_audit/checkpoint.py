"""Checkpoint provenance without importing the GPU stack."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path


def remap_lerobot_pi0_key(key: str) -> str | None:
    """Map LeRobot Pi0 checkpoint keys to Transformers Pi0 keys."""
    if "paligemma_with_expert.paligemma.lm_head" in key:
        return None
    replacements = (
        (
            "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.",
            "model.vlm.vision_tower.",
        ),
        (
            "model.paligemma_with_expert.paligemma.model.language_model.",
            "model.vlm.language_model.",
        ),
        (
            "model.paligemma_with_expert.paligemma.model.multi_modal_projector.",
            "model.vlm.multi_modal_projector.",
        ),
        ("model.paligemma_with_expert.gemma_expert.model.", "model.dit."),
        (
            "model.paligemma_with_expert.gemma_expert.lm_head.",
            "model.dit.embed_tokens.",
        ),
        ("model.action_in_proj.", "embed_action_time.action_in_proj."),
        ("model.state_proj.", "embed_action_time.state_proj."),
        ("model.action_time_mlp_in.", "embed_action_time.action_time_mlp_in."),
        ("model.action_time_mlp_out.", "embed_action_time.action_time_mlp_out."),
        ("model.action_out_proj.", "action_out_proj."),
    )
    for source, destination in replacements:
        key = key.replace(source, destination)
    return key


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(
    paths: Iterable[str | Path],
    *,
    relative_to: str | Path | None = None,
    uri_prefix: str | None = None,
) -> list[dict[str, str | int]]:
    """Hash checkpoint artifacts without loading tensors.

    ``relative_to`` removes machine-specific path prefixes from committed
    manifests. ``uri_prefix`` can be added after weights move to Hugging Face or
    object storage; local-only manifests intentionally omit it.
    """
    root = Path(relative_to).resolve() if relative_to is not None else None
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        rendered_path = path.resolve().relative_to(root).as_posix() if root else str(path)
        row: dict[str, str | int] = {
            "path": rendered_path,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if uri_prefix is not None:
            row["uri"] = f"{uri_prefix.rstrip('/')}/{rendered_path}"
        rows.append(row)
    return rows
