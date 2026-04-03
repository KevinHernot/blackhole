from __future__ import annotations

"""Distortion validation helpers for the Blackhole NumPy prototype."""

from dataclasses import dataclass

import numpy as np

from .metrics import ReconstructionStats, as_float_array, ensure_2d, reconstruction_stats

EPSILON = 1e-12


def max_abs_error(original: np.ndarray | list[float], reconstructed: np.ndarray | list[float]) -> float:
    original_array = as_float_array(original, name="original")
    reconstructed_array = as_float_array(reconstructed, name="reconstructed")
    if original_array.shape != reconstructed_array.shape:
        raise ValueError(
            "original and reconstructed must have the same shape: "
            f"{original_array.shape} != {reconstructed_array.shape}"
        )
    return float(np.max(np.abs(original_array - reconstructed_array))) if original_array.size else 0.0


def stable_softmax(logits: np.ndarray | list[float], axis: int = -1) -> np.ndarray:
    logits_array = as_float_array(logits, name="logits")
    shifted = logits_array - np.max(logits_array, axis=axis, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.maximum(np.sum(exp_logits, axis=axis, keepdims=True), EPSILON)


def kl_divergence(
    reference_logits: np.ndarray | list[float],
    candidate_logits: np.ndarray | list[float],
    *,
    axis: int = -1,
) -> float:
    reference_probs = stable_softmax(reference_logits, axis=axis)
    candidate_probs = stable_softmax(candidate_logits, axis=axis)
    if reference_probs.shape != candidate_probs.shape:
        raise ValueError(
            "reference_logits and candidate_logits must have the same shape after softmax: "
            f"{reference_probs.shape} != {candidate_probs.shape}"
        )
    log_ratio = np.log(np.maximum(reference_probs, EPSILON)) - np.log(np.maximum(candidate_probs, EPSILON))
    return float(np.mean(np.sum(reference_probs * log_ratio, axis=axis)))


def attention_output(
    query: np.ndarray | list[float],
    keys: np.ndarray | list[list[float]],
    values: np.ndarray | list[list[float]],
) -> np.ndarray:
    query_array = ensure_2d(query, name="query")
    if query_array.shape[0] != 1:
        raise ValueError("query must describe a single attention query vector.")

    key_array = ensure_2d(keys, name="keys")
    value_array = ensure_2d(values, name="values")
    if key_array.shape != value_array.shape:
        raise ValueError(f"keys and values must share the same shape: {key_array.shape} != {value_array.shape}")
    if query_array.shape[1] != key_array.shape[1]:
        raise ValueError(
            "query and keys must share the same feature dimension: "
            f"{query_array.shape[1]} != {key_array.shape[1]}"
        )

    scale = float(np.sqrt(key_array.shape[1]))
    scores = (key_array @ query_array[0]) / max(scale, EPSILON)
    weights = stable_softmax(scores, axis=0)
    return weights @ value_array


@dataclass(frozen=True)
class DistortionThresholds:
    mse_max: float | None = None
    rmse_max: float | None = None
    relative_l2_max: float | None = None
    mean_cosine_min: float | None = None
    max_abs_error_max: float | None = None
    kl_divergence_max: float | None = None


@dataclass(frozen=True)
class DistortionStats:
    reconstruction: ReconstructionStats
    max_abs_error: float
    kl_divergence: float | None = None


@dataclass(frozen=True)
class DistortionValidationResult:
    stats: DistortionStats
    passed: bool
    failures: tuple[str, ...]


def validate_reconstruction(
    original: np.ndarray | list[float],
    reconstructed: np.ndarray | list[float],
    *,
    thresholds: DistortionThresholds | None = None,
    reference_logits: np.ndarray | list[float] | None = None,
    candidate_logits: np.ndarray | list[float] | None = None,
) -> DistortionValidationResult:
    thresholds = thresholds or DistortionThresholds()
    reconstruction = reconstruction_stats(original, reconstructed)
    abs_error = max_abs_error(original, reconstructed)

    kl_value: float | None = None
    if reference_logits is not None and candidate_logits is not None:
        kl_value = kl_divergence(reference_logits, candidate_logits)

    failures: list[str] = []
    if thresholds.mse_max is not None and reconstruction.mse > thresholds.mse_max:
        failures.append(f"mse>{thresholds.mse_max}")
    if thresholds.rmse_max is not None and reconstruction.rmse > thresholds.rmse_max:
        failures.append(f"rmse>{thresholds.rmse_max}")
    if thresholds.relative_l2_max is not None and reconstruction.relative_l2 > thresholds.relative_l2_max:
        failures.append(f"relative_l2>{thresholds.relative_l2_max}")
    if thresholds.mean_cosine_min is not None and reconstruction.mean_cosine < thresholds.mean_cosine_min:
        failures.append(f"mean_cosine<{thresholds.mean_cosine_min}")
    if thresholds.max_abs_error_max is not None and abs_error > thresholds.max_abs_error_max:
        failures.append(f"max_abs_error>{thresholds.max_abs_error_max}")
    if thresholds.kl_divergence_max is not None and kl_value is not None and kl_value > thresholds.kl_divergence_max:
        failures.append(f"kl_divergence>{thresholds.kl_divergence_max}")

    return DistortionValidationResult(
        stats=DistortionStats(
            reconstruction=reconstruction,
            max_abs_error=abs_error,
            kl_divergence=kl_value,
        ),
        passed=not failures,
        failures=tuple(failures),
    )


def validate_attention_preservation(
    query: np.ndarray | list[float],
    reference_keys: np.ndarray | list[list[float]],
    reference_values: np.ndarray | list[list[float]],
    candidate_keys: np.ndarray | list[list[float]],
    candidate_values: np.ndarray | list[list[float]],
    *,
    thresholds: DistortionThresholds | None = None,
) -> DistortionValidationResult:
    reference_output = attention_output(query, reference_keys, reference_values)
    candidate_output = attention_output(query, candidate_keys, candidate_values)
    return validate_reconstruction(reference_output, candidate_output, thresholds=thresholds)


__all__ = [
    "DistortionStats",
    "DistortionThresholds",
    "DistortionValidationResult",
    "EPSILON",
    "attention_output",
    "kl_divergence",
    "max_abs_error",
    "stable_softmax",
    "validate_attention_preservation",
    "validate_reconstruction",
]
