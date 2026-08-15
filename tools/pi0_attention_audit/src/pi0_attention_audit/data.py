"""Episode readers for deployment captures and LeRobot datasets."""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import AuditConfig


@dataclass(frozen=True)
class FrameSample:
    episode: int
    frame: int
    image_bgr: NDArray[np.uint8]
    state: NDArray[np.float32] | None

    @property
    def sample_id(self) -> str:
        return f"ep{self.episode:06d}_f{self.frame:06d}"


def _nested(value: Any, keys: Sequence[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"missing state key path component {key!r}")
        current = current[key]
    return current


def _padded_state(values: Sequence[float], state_dim: int, num_joints: int) -> NDArray[np.float32]:
    state = np.zeros(state_dim, dtype=np.float32)
    joints = np.asarray(values, dtype=np.float32)
    if len(joints) < num_joints:
        raise ValueError(f"expected at least {num_joints} joints, found {len(joints)}")
    state[:num_joints] = joints[:num_joints]
    return state


def _read_frames(
    video_path: Path,
    frame_indices: Sequence[int],
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("OpenCV is required; install pi0-attention-audit[analysis]") from exc

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for requested in frame_indices:
            if requested < 0 or requested >= total:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
            ok, frame = capture.read()
            if ok:
                yield requested, frame
    finally:
        capture.release()


def iter_deployment_samples(
    config: AuditConfig,
    max_samples: int | None = None,
) -> Iterator[FrameSample]:
    """Yield deployment frames paired with joint state from trusted pickle files."""
    count = 0
    for episode in config.deployment.episodes:
        episode_dir = config.deployment.root / f"{episode:04d}"
        state_path = episode_dir / config.deployment.state_name
        # The source is a lab-generated capture. Do not use pickle input from an
        # untrusted party.
        with state_path.open("rb") as handle:
            state_payload = pickle.load(handle)  # noqa: S301
        states = _nested(state_payload, config.deployment.state_keys)
        indices = range(0, len(states), config.deployment.frame_stride)
        for frame_index, frame in _read_frames(
            episode_dir / config.deployment.video_name,
            indices,
        ):
            yield FrameSample(
                episode=episode,
                frame=frame_index,
                image_bgr=frame,
                state=_padded_state(
                    states[frame_index],
                    config.analysis.state_dim,
                    config.analysis.num_joints,
                ),
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def iter_lerobot_samples(
    config: AuditConfig,
    max_samples: int | None = None,
) -> Iterator[FrameSample]:
    """Yield LeRobot frames paired with `observation.state` from parquet."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("pandas and pyarrow are required for LeRobot input") from exc

    count = 0
    for episode in config.lerobot.episodes:
        parquet_path = config.lerobot.parquet_dir / f"episode_{episode:06d}.parquet"
        frame_table = pd.read_parquet(parquet_path, columns=[config.lerobot.state_column])
        states = frame_table[config.lerobot.state_column].tolist()
        indices = range(0, len(states), config.lerobot.frame_stride)
        video_path = config.lerobot.video_dir / f"episode_{episode:06d}.mp4"
        for frame_index, frame in _read_frames(video_path, indices):
            yield FrameSample(
                episode=episode,
                frame=frame_index,
                image_bgr=frame,
                state=_padded_state(
                    states[frame_index],
                    config.analysis.state_dim,
                    config.analysis.num_joints,
                ),
            )
            count += 1
            if max_samples is not None and count >= max_samples:
                return


def iter_siglip_samples(
    config: AuditConfig,
    max_samples: int | None = None,
) -> Iterator[FrameSample]:
    count = 0
    for episode in config.siglip.episodes:
        video_path = config.lerobot.video_dir / f"episode_{episode:06d}.mp4"
        for frame_index, frame in _read_frames(video_path, config.siglip.frame_indices):
            yield FrameSample(episode=episode, frame=frame_index, image_bgr=frame, state=None)
            count += 1
            if max_samples is not None and count >= max_samples:
                return
