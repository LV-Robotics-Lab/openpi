"""Attention extraction algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import AnalysisConfig
from .metrics import normalize_heatmap

ACTION_EXPERT_METHOD_ID = "action-expert-v3-real-state-multistep-raw"


def action_expert_attention(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    pixel_values: Any,
    state: Any,
    analysis: AnalysisConfig,
    initial_noise: Any,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Run the canonical v3 action-expert attention method.

    The historical ``action_expert_attention_v2.py`` scratch file was edited in
    place after its v2 results were produced. Its current code uses raw softmax
    weights for region metrics, which the archived report calls v3. This
    implementation makes that semantic version explicit while keeping the CLI
    command name stable.
    """
    import torch

    device = pixel_values.device
    pixel_attention_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
    delta = 1.0 / analysis.num_denoising_steps
    noisy_actions = initial_noise.clone()
    heatmaps = []
    captured: list[Any | None] = [None]

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured[0] = output[1].detach().float()

    handle = model.model.dit.layers[-1].self_attn.register_forward_hook(hook)
    try:
        for step in range(analysis.num_denoising_steps):
            captured[0] = None
            timestep = torch.tensor(
                [1.0 - step * delta],
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                output = model(
                    state=state,
                    noise=noisy_actions,
                    timestep=timestep,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    attention_mask=attention_mask,
                )
            noisy_actions = noisy_actions - output.logits * delta
            if captured[0] is None:
                continue
            image_attention = captured[0][0, :, :, : analysis.num_image_patches]
            heatmap = image_attention.mean(dim=0).mean(dim=0).cpu().numpy()
            heatmaps.append(heatmap.reshape(analysis.patch_grid, analysis.patch_grid))
    finally:
        handle.remove()

    if not heatmaps:
        raise RuntimeError("no action-expert attention was captured")
    raw = np.asarray(np.mean(heatmaps, axis=0), dtype=np.float32)
    return raw, normalize_heatmap(raw)


def siglip_attention_rollout(model: Any, pixel_values: Any) -> NDArray[np.float32]:
    """Compute residual attention rollout across every SigLIP encoder layer."""
    import torch

    captured = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            captured.append(output[1].squeeze(0).mean(0).detach().float())

    handles = [layer.self_attn.register_forward_hook(hook) for layer in model.encoder.layers]
    try:
        with torch.no_grad():
            model(pixel_values=pixel_values)
    finally:
        for handle in handles:
            handle.remove()
    if not captured:
        raise RuntimeError("no SigLIP attention was captured")

    rollout = None
    identity = torch.eye(captured[0].shape[-1], device=pixel_values.device)
    for attention in captured:
        residual = attention + identity
        residual = residual / residual.sum(dim=-1, keepdim=True)
        rollout = residual if rollout is None else residual @ rollout
    importance = rollout.mean(dim=0).cpu().numpy()
    side = int(round(np.sqrt(importance.size)))
    if side * side != importance.size:
        raise RuntimeError(f"SigLIP token count is not square: {importance.size}")
    return normalize_heatmap(importance.reshape(side, side))
