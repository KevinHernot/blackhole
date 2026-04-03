from __future__ import annotations

"""Explicit outlier-channel strategy for Blackhole tensors."""

from dataclasses import dataclass

import numpy as np

from .metrics import ReconstructionStats, ensure_2d, reconstruction_stats

_ALLOWED_SCORE_METRICS = {"l2", "max_abs", "std"}


def score_outlier_channels(
    values: np.ndarray | list[list[float]],
    *,
    metric: str = "max_abs",
) -> np.ndarray:
    value_array = ensure_2d(values, name="values")
    if metric not in _ALLOWED_SCORE_METRICS:
        raise ValueError(f"metric must be one of {_ALLOWED_SCORE_METRICS}, got {metric!r}.")

    if metric == "max_abs":
        return np.max(np.abs(value_array), axis=0)
    if metric == "std":
        return np.std(value_array, axis=0)
    return np.linalg.norm(value_array, axis=0) / np.sqrt(max(value_array.shape[0], 1))


@dataclass(frozen=True)
class OutlierChannelStrategy:
    score_metric: str = "max_abs"
    zscore_threshold: float = 2.5
    max_outlier_fraction: float = 0.125
    min_outlier_channels: int = 1


@dataclass(frozen=True)
class OutlierChannelSplit:
    base_values: np.ndarray
    outlier_indices: np.ndarray
    outlier_values: np.ndarray
    channel_scores: np.ndarray
    threshold_score: float
    original_shape: tuple[int, ...]

    def reconstruct(self) -> np.ndarray:
        reconstructed = self.base_values.copy()
        if self.outlier_indices.size:
            reconstructed[:, self.outlier_indices] = self.outlier_values
        if len(self.original_shape) == 1:
            return reconstructed[0]
        return reconstructed

    @property
    def outlier_fraction(self) -> float:
        channel_count = max(self.channel_scores.shape[0], 1)
        return float(self.outlier_indices.size / channel_count)


@dataclass(frozen=True)
class OutlierChannelStats:
    outlier_count: int
    outlier_fraction: float
    threshold_score: float
    outlier_energy_fraction: float
    base_reconstruction: ReconstructionStats
    restored_reconstruction: ReconstructionStats


def select_outlier_channels(
    values: np.ndarray | list[list[float]],
    *,
    strategy: OutlierChannelStrategy | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    strategy = strategy or OutlierChannelStrategy()
    scores = score_outlier_channels(values, metric=strategy.score_metric)
    channel_count = scores.shape[0]
    if channel_count == 0:
        return np.array([], dtype=int), scores, float("inf")

    max_channels = int(np.ceil(channel_count * strategy.max_outlier_fraction))
    max_channels = max(max_channels, strategy.min_outlier_channels)
    max_channels = min(max_channels, channel_count)

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    threshold_score = mean + strategy.zscore_threshold * std if std > 0 else float(np.max(scores)) + 1.0
    selected = np.flatnonzero(scores >= threshold_score)

    if selected.size < strategy.min_outlier_channels:
        ordering = np.argsort(scores)[::-1]
        selected = ordering[: strategy.min_outlier_channels]

    if max_channels > 0 and selected.size > max_channels:
        ordering = selected[np.argsort(scores[selected])[::-1]]
        selected = ordering[:max_channels]

    selected = np.sort(selected.astype(int, copy=False))
    if selected.size:
        threshold_score = float(np.min(scores[selected]))

    return selected, scores, threshold_score


def split_outlier_channels(
    values: np.ndarray | list[list[float]],
    *,
    strategy: OutlierChannelStrategy | None = None,
) -> OutlierChannelSplit:
    value_array = ensure_2d(values, name="values")
    selected, scores, threshold_score = select_outlier_channels(value_array, strategy=strategy)
    base_values = value_array.copy()
    outlier_values = base_values[:, selected].copy() if selected.size else np.zeros((value_array.shape[0], 0), dtype=float)
    if selected.size:
        base_values[:, selected] = 0.0

    original_shape = np.asarray(values, dtype=float).shape
    if len(original_shape) == 0:
        original_shape = (1,)

    return OutlierChannelSplit(
        base_values=base_values,
        outlier_indices=selected,
        outlier_values=outlier_values,
        channel_scores=scores,
        threshold_score=threshold_score,
        original_shape=tuple(original_shape),
    )


def restore_outlier_channels(split: OutlierChannelSplit) -> np.ndarray:
    return split.reconstruct()


def outlier_channel_stats(
    original: np.ndarray | list[list[float]],
    split: OutlierChannelSplit,
) -> OutlierChannelStats:
    original_array = ensure_2d(original, name="original")
    base_reconstruction = reconstruction_stats(original_array, split.base_values)
    restored_reconstruction = reconstruction_stats(original_array, ensure_2d(split.reconstruct(), name="reconstructed"))

    total_energy = float(np.linalg.norm(original_array.ravel()))
    outlier_energy = float(np.linalg.norm(split.outlier_values.ravel()))
    outlier_energy_fraction = outlier_energy / max(total_energy, 1e-12)

    return OutlierChannelStats(
        outlier_count=int(split.outlier_indices.size),
        outlier_fraction=split.outlier_fraction,
        threshold_score=split.threshold_score,
        outlier_energy_fraction=outlier_energy_fraction,
        base_reconstruction=base_reconstruction,
        restored_reconstruction=restored_reconstruction,
    )


__all__ = [
    "OutlierChannelSplit",
    "OutlierChannelStats",
    "OutlierChannelStrategy",
    "outlier_channel_stats",
    "restore_outlier_channels",
    "score_outlier_channels",
    "select_outlier_channels",
    "split_outlier_channels",
]
