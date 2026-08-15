"""Lazy model loading and preprocessing for Pi0 attention extraction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .checkpoint import remap_lerobot_pi0_key
from .config import AnalysisConfig

SIGLIP_WEIGHT_PREFIX = "model.paligemma_with_expert.paligemma.model.vision_tower.vision_model."


class MissingAnalysisDependency(RuntimeError):
    pass


def _runtime() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import cv2
        import safetensors.torch as safetensors_torch
        import torch
        from transformers import PI0Config, PI0ForConditionalGeneration, SiglipVisionConfig
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    except ImportError as exc:  # pragma: no cover - optional environment
        raise MissingAnalysisDependency(
            "GPU analysis dependencies are missing; install pi0-attention-audit[analysis]"
        ) from exc
    return (
        torch,
        cv2,
        safetensors_torch,
        PI0Config,
        PI0ForConditionalGeneration,
        (SiglipVisionConfig, SiglipVisionModel),
    )


def resolve_device(requested: str | None = None) -> Any:
    torch, *_ = _runtime()
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pi0(checkpoint_path: str, analysis: AnalysisConfig, device: Any) -> Any:
    torch, _, safetensors_torch, PI0Config, PI0ForConditionalGeneration, _ = _runtime()
    raw = safetensors_torch.load_file(checkpoint_path, device="cpu")
    remapped = {}
    for key, value in raw.items():
        mapped = remap_lerobot_pi0_key(key)
        if mapped is not None:
            remapped[mapped] = value

    config = PI0Config(
        max_action_dim=analysis.action_dim,
        max_state_dim=analysis.state_dim,
        chunk_size=analysis.chunk_size,
    )
    config.dit_config.attn_implementation = "eager"
    config.dit_config._attn_implementation = "eager"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = PI0ForConditionalGeneration(config).to(dtype=dtype)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    real_missing = [key for key in missing if "lm_head" not in key]
    if real_missing:
        raise RuntimeError(f"checkpoint is missing required Pi0 keys: {real_missing[:5]}")
    if unexpected:
        print(f"warning: ignored {len(unexpected)} unexpected checkpoint keys")
    return model.to(device).eval()


def load_siglip(checkpoint_path: str, device: Any) -> Any:
    torch, _, safetensors_torch, _, _, siglip_types = _runtime()
    SiglipVisionConfig, SiglipVisionModel = siglip_types
    raw = safetensors_torch.load_file(checkpoint_path, device="cpu")
    weights = {
        key[len(SIGLIP_WEIGHT_PREFIX) :]: value
        for key, value in raw.items()
        if key.startswith(SIGLIP_WEIGHT_PREFIX)
    }
    config = SiglipVisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
        vision_use_head=False,
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = SiglipVisionModel._from_config(config, attn_implementation="eager").to(dtype=dtype)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    real_missing = [key for key in missing if "head" not in key]
    if real_missing:
        raise RuntimeError(f"checkpoint is missing required SigLIP keys: {real_missing[:5]}")
    if unexpected:
        print(f"warning: ignored {len(unexpected)} unexpected SigLIP keys")
    return model.to(device).eval()


def build_input_ids(tokenizer: Any, analysis: AnalysisConfig, device: Any) -> tuple[Any, Any]:
    torch, *_ = _runtime()
    text_ids = tokenizer(analysis.task_prompt, add_special_tokens=False)["input_ids"]
    ids = [analysis.image_token_id] * analysis.num_image_patches
    ids += [tokenizer.bos_token_id] + text_ids
    max_length = analysis.num_image_patches + analysis.tokenizer_max_length
    real_count = min(len(ids), max_length)
    pad_id = tokenizer.pad_token_id or 0
    ids = ids[:max_length] + [pad_id] * max(0, max_length - len(ids))
    mask = [1] * real_count + [0] * (max_length - real_count)
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.long, device=device),
    )


def frame_to_tensor(
    frame_bgr: NDArray[np.uint8],
    device: Any,
    dtype: Any,
    add_camera_axis: bool,
) -> Any:
    torch, cv2, *_ = _runtime()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).to(device=device, dtype=dtype).div(255.0)
    tensor = ((tensor - 0.5) / 0.5).permute(2, 0, 1).unsqueeze(0)
    return tensor.unsqueeze(0) if add_camera_axis else tensor
