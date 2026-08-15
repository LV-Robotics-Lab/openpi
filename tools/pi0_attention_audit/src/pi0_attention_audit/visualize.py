"""Compact evidence renders for attention comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def overlay_heatmap(
    frame_bgr: NDArray[np.uint8],
    attention: NDArray[np.floating],
    alpha: float = 0.55,
) -> NDArray[np.uint8]:
    import cv2

    height, width = frame_bgr.shape[:2]
    heatmap = cv2.resize(attention, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0.0)


def save_comparison_grid(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    figure, axes = plt.subplots(len(rows), 4, figsize=(16, 3.5 * len(rows)))
    if len(rows) == 1:
        axes = axes[np.newaxis, :]
    for index, row in enumerate(rows):
        frame = row["frame"]
        with_hand = row["with_hand"]
        without_hand = row["without_hand"]
        difference = with_hand - without_hand
        panels = (
            (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), "Original", None),
            (
                cv2.cvtColor(overlay_heatmap(frame, with_hand), cv2.COLOR_BGR2RGB),
                "With-hand policy",
                None,
            ),
            (
                cv2.cvtColor(overlay_heatmap(frame, without_hand), cv2.COLOR_BGR2RGB),
                "No-hand policy",
                None,
            ),
            (difference, "With-hand minus no-hand", "RdBu_r"),
        )
        for column, (image, title, color_map) in enumerate(panels):
            axes[index, column].imshow(image, cmap=color_map, vmin=-0.15, vmax=0.15)
            axes[index, column].set_title(title, fontsize=8)
            axes[index, column].axis("off")
        axes[index, 0].set_ylabel(str(row["sample_id"]), fontsize=7, rotation=0, labelpad=55)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(figure)


def save_region_chart(
    with_hand: Mapping[str, Sequence[float]],
    without_hand: Mapping[str, Sequence[float]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(with_hand)
    with_mean = [float(np.mean(with_hand[name])) * 100 for name in names]
    without_mean = [float(np.mean(without_hand[name])) * 100 for name in names]
    x_values = np.arange(len(names))
    width = 0.36
    figure, (left, right) = plt.subplots(1, 2, figsize=(14, 5))
    left.bar(x_values - width / 2, with_mean, width, label="With-hand policy")
    left.bar(x_values + width / 2, without_mean, width, label="No-hand policy")
    left.set_xticks(x_values)
    left.set_xticklabels(names, rotation=20, ha="right")
    left.set_ylabel("Relative image-attention fraction (%)")
    left.legend()
    differences = [a - b for a, b in zip(with_mean, without_mean, strict=True)]
    right.bar(x_values, differences)
    right.axhline(0.0, color="black", linewidth=0.8)
    right.set_xticks(x_values)
    right.set_xticklabels(names, rotation=20, ha="right")
    right.set_ylabel("Difference (percentage points)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
