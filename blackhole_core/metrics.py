from __future__ import annotations

"""Shared numeric helpers for real Blackhole NumPy prototypes."""

from dataclasses import dataclass

import numpy as np

EPSILON = 1e-12


def as_float_array(values: np.ndarray | list[float], name: str = "values") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension.")
    return array


def ensure_2d(values: np.ndarray | list[float], name: str = "values") -> np.ndarray:
    array = as_float_array(values, name=name)
    if array.ndim == 1:
        return array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {array.shape}.")
    return array


def l2_normalize(values: np.ndarray | list[float], axis: int = -1, eps: float = EPSILON) -> np.ndarray:
    array = as_float_array(values)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_similarity(
    left: np.ndarray | list[float],
    right: np.ndarray | list[float],
    axis: int = -1,
) -> np.ndarray:
    left_norm = l2_normalize(left, axis=axis)
    right_norm = l2_normalize(right, axis=axis)
    return np.sum(left_norm * right_norm, axis=axis)


def pairwise_cosine_scores(query: np.ndarray | list[float], matrix: np.ndarray | list[float]) -> np.ndarray:
    query_array = ensure_2d(query, name="query")
    if query_array.shape[0] != 1:
        raise ValueError("query must describe a single vector.")

    matrix_array = ensure_2d(matrix, name="matrix")
    if query_array.shape[1] != matrix_array.shape[1]:
        raise ValueError(
            "query and matrix must have the same feature dimension: "
            f"{query_array.shape[1]} != {matrix_array.shape[1]}"
        )

    normalized_query = l2_normalize(query_array)[0]
    normalized_matrix = l2_normalize(matrix_array)
    return normalized_matrix @ normalized_query


def mean_squared_error(original: np.ndarray | list[float], reconstructed: np.ndarray | list[float]) -> float:
    original_array = as_float_array(original, name="original")
    reconstructed_array = as_float_array(reconstructed, name="reconstructed")
    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            "original and reconstructed must have the same shape: "
            f"{original_array.shape} != {reconstructed_array.shape}"
        )
    return float(np.mean((original_array - reconstructed_array) ** 2))


def root_mean_squared_error(original: np.ndarray | list[float], reconstructed: np.ndarray | list[float]) -> float:
    return float(np.sqrt(mean_squared_error(original, reconstructed)))


def relative_l2_error(original: np.ndarray | list[float], reconstructed: np.ndarray | list[float]) -> float:
    original_array = as_float_array(original, name="original")
    reconstructed_array = as_float_array(reconstructed, name="reconstructed")
    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            "original and reconstructed must have the same shape: "
            f"{original_array.shape} != {reconstructed_array.shape}"
        )
    numerator = np.linalg.norm((original_array - reconstructed_array).ravel())
    denominator = max(np.linalg.norm(original_array.ravel()), EPSILON)
    return float(numerator / denominator)


def mean_cosine_similarity(original: np.ndarray | list[float], reconstructed: np.ndarray | list[float]) -> float:
    original_array = as_float_array(original, name="original")
    reconstructed_array = as_float_array(reconstructed, name="reconstructed")
    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            "original and reconstructed must have the same shape: "
            f"{original_array.shape} != {reconstructed_array.shape}"
        )
    if original_array.ndim == 1:
        original_array = original_array[np.newaxis, :]
        reconstructed_array = reconstructed_array[np.newaxis, :]
    elif original_array.ndim > 2:
        original_array = original_array.reshape(-1, original_array.shape[-1])
        reconstructed_array = reconstructed_array.reshape(-1, reconstructed_array.shape[-1])
    return float(np.mean(cosine_similarity(original_array, reconstructed_array)))


@dataclass(frozen=True)
class ReconstructionStats:
    mse: float
    rmse: float
    mean_cosine: float
    relative_l2: float


def reconstruction_stats(
    original: np.ndarray | list[float],
    reconstructed: np.ndarray | list[float],
) -> ReconstructionStats:
    return ReconstructionStats(
        mse=mean_squared_error(original, reconstructed),
        rmse=root_mean_squared_error(original, reconstructed),
        mean_cosine=mean_cosine_similarity(original, reconstructed),
        relative_l2=relative_l2_error(original, reconstructed),
    )


__all__ = [
    "EPSILON",
    "ReconstructionStats",
    "as_float_array",
    "cosine_similarity",
    "ensure_2d",
    "l2_normalize",
    "mean_cosine_similarity",
    "mean_squared_error",
    "pairwise_cosine_scores",
    "reconstruction_stats",
    "relative_l2_error",
    "root_mean_squared_error",
]
