"""Pure numerical helpers shared by visual and action attention audits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .config import BBox


def normalize_heatmap(values: NDArray[np.floating]) -> NDArray[np.float32]:
    array = np.asarray(values, dtype=np.float32)
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.float32)
    return (array - low) / (high - low)


def region_view(values: NDArray[np.floating], bbox: BBox) -> NDArray[np.floating]:
    height, width = values.shape
    x1, y1, x2, y2 = bbox
    return values[
        int(y1 * height) : int(y2 * height),
        int(x1 * width) : int(x2 * width),
    ]


def region_fraction(values: NDArray[np.floating], bbox: BBox) -> float:
    """Fraction of raw image attention that falls inside a region."""
    array = np.asarray(values)
    total = float(array.sum())
    if total <= 0:
        return 0.0
    return float(region_view(array, bbox).sum() / total)


def region_mean(values: NDArray[np.floating], bbox: BBox) -> float:
    """Mean display-normalized attention inside a region."""
    selected = region_view(np.asarray(values), bbox)
    return float(selected.mean()) if selected.size else 0.0


def summarize_regions(
    samples: Sequence[Mapping[str, float]], region_names: Sequence[str]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in region_names:
        values = np.asarray([float(sample[name]) for sample in samples], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
            "count": int(len(values)),
        }
    return result
