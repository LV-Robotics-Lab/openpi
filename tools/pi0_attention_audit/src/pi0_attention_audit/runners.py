"""End-to-end analysis runners used by the command-line interface."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .attention import (
    ACTION_EXPERT_METHOD_ID,
    action_expert_attention,
    siglip_attention_rollout,
)
from .config import AuditConfig
from .data import FrameSample, iter_deployment_samples, iter_lerobot_samples, iter_siglip_samples
from .metrics import region_fraction, region_mean
from .modeling import (
    build_input_ids,
    frame_to_tensor,
    load_pi0,
    load_siglip,
    resolve_device,
)
from .visualize import save_comparison_grid, save_region_chart

ActionSource = Literal["deployment", "lerobot"]


def _check_paths(config: AuditConfig, source: str) -> None:
    missing = [check for check in config.path_checks(source) if not check.exists]  # type: ignore[arg-type]
    if missing:
        formatted = "\n".join(f"- {check.label}: {check.path}" for check in missing)
        raise FileNotFoundError(f"runtime inputs are missing:\n{formatted}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _action_samples(
    config: AuditConfig,
    source: ActionSource,
    max_samples: int | None,
) -> Iterator[FrameSample]:
    if source == "deployment":
        return iter_deployment_samples(config, max_samples=max_samples)
    return iter_lerobot_samples(config, max_samples=max_samples)


def run_action_expert(
    config: AuditConfig,
    source: ActionSource,
    max_samples: int | None = None,
    device_name: str | None = None,
) -> Path:
    """Compare action-expert attention with matched noise for both policies."""
    _check_paths(config, source)
    import torch
    from transformers import AutoTokenizer

    device = resolve_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(config.models.tokenizer)
    with_hand_model = load_pi0(str(config.models.with_hand), config.analysis, device)
    without_hand_model = load_pi0(str(config.models.without_hand), config.analysis, device)
    input_ids, attention_mask = build_input_ids(tokenizer, config.analysis, device)

    output = config.output_root / f"action_expert_{source}"
    output.mkdir(parents=True, exist_ok=True)
    accumulators = {
        "with_hand": {name: [] for name in config.regions},
        "without_hand": {name: [] for name in config.regions},
    }
    rows = []
    samples = []

    for sample_index, sample in enumerate(_action_samples(config, source, max_samples)):
        if sample.state is None:
            raise RuntimeError(f"sample {sample.sample_id} has no state")
        state = (
            torch.from_numpy(sample.state)
            .unsqueeze(0)
            .to(
                device=device,
                dtype=with_hand_model.dtype,
            )
        )
        pixels = frame_to_tensor(
            sample.image_bgr,
            device=device,
            dtype=with_hand_model.dtype,
            add_camera_axis=True,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(config.analysis.random_seed + sample_index)
        shared_noise = torch.randn(
            1,
            config.analysis.chunk_size,
            config.analysis.action_dim,
            generator=generator,
            device=device,
            dtype=with_hand_model.dtype,
        )
        raw_with, visual_with = action_expert_attention(
            with_hand_model,
            input_ids,
            attention_mask,
            pixels,
            state,
            config.analysis,
            shared_noise,
        )
        raw_without, visual_without = action_expert_attention(
            without_hand_model,
            input_ids,
            attention_mask,
            pixels,
            state.to(dtype=without_hand_model.dtype),
            config.analysis,
            shared_noise.to(dtype=without_hand_model.dtype),
        )
        sample_stats = {
            "sample_id": sample.sample_id,
            "episode": sample.episode,
            "frame": sample.frame,
            "with_hand": {},
            "without_hand": {},
            "difference": {},
        }
        for name, bbox in config.regions.items():
            with_value = region_fraction(raw_with, bbox)
            without_value = region_fraction(raw_without, bbox)
            sample_stats["with_hand"][name] = with_value
            sample_stats["without_hand"][name] = without_value
            sample_stats["difference"][name] = with_value - without_value
            accumulators["with_hand"][name].append(with_value)
            accumulators["without_hand"][name].append(without_value)
        samples.append(sample_stats)
        if len(rows) < 16:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "frame": sample.image_bgr,
                    "with_hand": visual_with,
                    "without_hand": visual_without,
                }
            )

    if not samples:
        raise RuntimeError("no samples were produced")
    payload = {
        "method": {
            "id": ACTION_EXPERT_METHOD_ID,
            "source": source,
            "attention": "last DiT layer action queries to image keys",
            "denoising_steps": config.analysis.num_denoising_steps,
            "state": "real joint state zero-padded to configured state_dim",
            "noise": "same seeded initial noise reused for both policy variants",
            "random_seed": config.analysis.random_seed,
            "region_metric": "raw region sum divided by all image-patch attention",
        },
        "samples": samples,
        "region_summary": {
            model: {
                name: {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": len(values),
                }
                for name, values in regions.items()
            }
            for model, regions in accumulators.items()
        },
    }
    _write_json(output / "action_stats.json", payload)
    save_comparison_grid(rows, output / "action_comparison.png")
    save_region_chart(
        accumulators["with_hand"],
        accumulators["without_hand"],
        output / "action_region_stats.png",
    )
    return output


def run_siglip(
    config: AuditConfig,
    max_samples: int | None = None,
    device_name: str | None = None,
) -> Path:
    """Compare display-normalized SigLIP attention rollout."""
    _check_paths(config, "siglip")
    device = resolve_device(device_name)
    with_hand_model = load_siglip(str(config.models.with_hand), device)
    without_hand_model = load_siglip(str(config.models.without_hand), device)
    output = config.output_root / "siglip"
    output.mkdir(parents=True, exist_ok=True)
    accumulators = {
        "with_hand": {name: [] for name in config.regions},
        "without_hand": {name: [] for name in config.regions},
    }
    rows = []
    samples = []
    for sample in iter_siglip_samples(config, max_samples=max_samples):
        pixels_with = frame_to_tensor(
            sample.image_bgr,
            device=device,
            dtype=with_hand_model.dtype,
            add_camera_axis=False,
        )
        pixels_without = pixels_with.to(dtype=without_hand_model.dtype)
        attention_with = siglip_attention_rollout(with_hand_model, pixels_with)
        attention_without = siglip_attention_rollout(without_hand_model, pixels_without)
        sample_stats = {
            "sample_id": sample.sample_id,
            "episode": sample.episode,
            "frame": sample.frame,
            "with_hand": {},
            "without_hand": {},
            "difference": {},
        }
        for name, bbox in config.regions.items():
            with_value = region_mean(attention_with, bbox)
            without_value = region_mean(attention_without, bbox)
            sample_stats["with_hand"][name] = with_value
            sample_stats["without_hand"][name] = without_value
            sample_stats["difference"][name] = with_value - without_value
            accumulators["with_hand"][name].append(with_value)
            accumulators["without_hand"][name].append(without_value)
        samples.append(sample_stats)
        if len(rows) < 16:
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "frame": sample.image_bgr,
                    "with_hand": attention_with,
                    "without_hand": attention_without,
                }
            )
    if not samples:
        raise RuntimeError("no samples were produced")
    _write_json(
        output / "siglip_stats.json",
        {
            "method": {
                "attention": "27-layer SigLIP residual attention rollout",
                "region_metric": "mean of display-normalized rollout within each region",
                "warning": "normalized attention is comparative diagnostic evidence, not causality",
            },
            "samples": samples,
        },
    )
    save_comparison_grid(rows, output / "siglip_comparison.png")
    save_region_chart(
        accumulators["with_hand"],
        accumulators["without_hand"],
        output / "siglip_region_stats.png",
    )
    return output
