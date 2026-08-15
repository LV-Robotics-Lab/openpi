"""Typed configuration and path preflight for attention audits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tomllib

BBox = tuple[float, float, float, float]
SourceName = Literal["deployment", "lerobot", "siglip"]


def _expand(value: str) -> str:
    return os.path.expanduser(os.path.expandvars(value))


def _path(value: str) -> Path:
    return Path(_expand(value))


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{name}] section")
    return value


@dataclass(frozen=True)
class ModelConfig:
    with_hand: Path
    without_hand: Path
    tokenizer: str


@dataclass(frozen=True)
class AnalysisConfig:
    task_prompt: str
    state_dim: int
    action_dim: int
    chunk_size: int
    num_joints: int
    num_image_patches: int
    patch_grid: int
    num_denoising_steps: int
    image_token_id: int
    tokenizer_max_length: int
    random_seed: int


@dataclass(frozen=True)
class DeploymentConfig:
    root: Path
    episodes: tuple[int, ...]
    frame_stride: int
    video_name: str
    state_name: str
    state_keys: tuple[str, ...]


@dataclass(frozen=True)
class LeRobotConfig:
    root: Path
    episodes: tuple[int, ...]
    frame_stride: int
    video_subdir: Path
    parquet_subdir: Path
    state_column: str

    @property
    def video_dir(self) -> Path:
        return self.root / self.video_subdir

    @property
    def parquet_dir(self) -> Path:
        return self.root / self.parquet_subdir


@dataclass(frozen=True)
class SiglipConfig:
    episodes: tuple[int, ...]
    frame_indices: tuple[int, ...]


@dataclass(frozen=True)
class PathCheck:
    label: str
    path: Path
    exists: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {"label": self.label, "path": str(self.path), "exists": self.exists}


@dataclass(frozen=True)
class AuditConfig:
    models: ModelConfig
    analysis: AnalysisConfig
    regions: dict[str, BBox]
    deployment: DeploymentConfig
    lerobot: LeRobotConfig
    siglip: SiglipConfig
    output_root: Path

    def path_checks(self, source: SourceName) -> list[PathCheck]:
        checks = [
            PathCheck("models.with_hand", self.models.with_hand, self.models.with_hand.is_file()),
            PathCheck(
                "models.without_hand",
                self.models.without_hand,
                self.models.without_hand.is_file(),
            ),
        ]
        if source == "deployment":
            for episode in self.deployment.episodes:
                episode_dir = self.deployment.root / f"{episode:04d}"
                checks.extend(
                    [
                        PathCheck(
                            f"deployment.episode_{episode}.video",
                            episode_dir / self.deployment.video_name,
                            (episode_dir / self.deployment.video_name).is_file(),
                        ),
                        PathCheck(
                            f"deployment.episode_{episode}.state",
                            episode_dir / self.deployment.state_name,
                            (episode_dir / self.deployment.state_name).is_file(),
                        ),
                    ]
                )
        elif source in {"lerobot", "siglip"}:
            episodes = self.lerobot.episodes if source == "lerobot" else self.siglip.episodes
            for episode in episodes:
                video = self.lerobot.video_dir / f"episode_{episode:06d}.mp4"
                checks.append(
                    PathCheck(f"{source}.episode_{episode}.video", video, video.is_file())
                )
                if source == "lerobot":
                    parquet = self.lerobot.parquet_dir / f"episode_{episode:06d}.parquet"
                    checks.append(
                        PathCheck(f"lerobot.episode_{episode}.parquet", parquet, parquet.is_file())
                    )
        else:  # pragma: no cover - guarded by argparse/type contract
            raise ValueError(f"unsupported source: {source}")
        return checks


def _validate_regions(regions: dict[str, BBox]) -> None:
    if not regions:
        raise ValueError("[regions] must define at least one bounding box")
    for name, (x1, y1, x2, y2) in regions.items():
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            raise ValueError(f"region {name!r} must stay within [0, 1]")
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"region {name!r} must have positive width and height")


def load_config(path: str | Path) -> AuditConfig:
    """Load a TOML config, expanding environment variables in path fields."""
    config_path = Path(path)
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    models = _section(data, "models")
    analysis = _section(data, "analysis")
    deployment = _section(data, "deployment")
    lerobot = _section(data, "lerobot")
    siglip = _section(data, "siglip")
    output = _section(data, "output")

    regions_raw = _section(data, "regions")
    regions: dict[str, BBox] = {}
    for name, values in regions_raw.items():
        if not isinstance(values, list) or len(values) != 4:
            raise ValueError(f"region {name!r} must contain four numbers")
        regions[name] = tuple(float(value) for value in values)  # type: ignore[assignment]
    _validate_regions(regions)

    result = AuditConfig(
        models=ModelConfig(
            with_hand=_path(str(models["with_hand"])),
            without_hand=_path(str(models["without_hand"])),
            tokenizer=str(models["tokenizer"]),
        ),
        analysis=AnalysisConfig(
            task_prompt=str(analysis["task_prompt"]),
            state_dim=int(analysis["state_dim"]),
            action_dim=int(analysis["action_dim"]),
            chunk_size=int(analysis["chunk_size"]),
            num_joints=int(analysis["num_joints"]),
            num_image_patches=int(analysis["num_image_patches"]),
            patch_grid=int(analysis["patch_grid"]),
            num_denoising_steps=int(analysis["num_denoising_steps"]),
            image_token_id=int(analysis["image_token_id"]),
            tokenizer_max_length=int(analysis["tokenizer_max_length"]),
            random_seed=int(analysis["random_seed"]),
        ),
        regions=regions,
        deployment=DeploymentConfig(
            root=_path(str(deployment["root"])),
            episodes=tuple(int(value) for value in deployment["episodes"]),
            frame_stride=int(deployment["frame_stride"]),
            video_name=str(deployment["video_name"]),
            state_name=str(deployment["state_name"]),
            state_keys=tuple(str(value) for value in deployment["state_keys"]),
        ),
        lerobot=LeRobotConfig(
            root=_path(str(lerobot["root"])),
            episodes=tuple(int(value) for value in lerobot["episodes"]),
            frame_stride=int(lerobot["frame_stride"]),
            video_subdir=Path(str(lerobot["video_subdir"])),
            parquet_subdir=Path(str(lerobot["parquet_subdir"])),
            state_column=str(lerobot["state_column"]),
        ),
        siglip=SiglipConfig(
            episodes=tuple(int(value) for value in siglip["episodes"]),
            frame_indices=tuple(int(value) for value in siglip["frame_indices"]),
        ),
        output_root=_path(str(output["root"])),
    )

    expected_patches = result.analysis.patch_grid**2
    if expected_patches != result.analysis.num_image_patches:
        raise ValueError(
            "analysis.patch_grid squared must equal analysis.num_image_patches "
            f"({expected_patches} != {result.analysis.num_image_patches})"
        )
    if result.analysis.num_joints > result.analysis.state_dim:
        raise ValueError("analysis.num_joints cannot exceed analysis.state_dim")
    if result.deployment.frame_stride <= 0 or result.lerobot.frame_stride <= 0:
        raise ValueError("frame strides must be positive")
    return result
