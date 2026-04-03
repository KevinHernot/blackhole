from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core import (
    DistortionThresholds,
    OutlierChannelStrategy,
    QualityValidationThresholds,
    TensorBundle,
    compare_tensor_bundles,
    outlier_channel_stats,
    split_outlier_channels,
    validate_attention_preservation,
    validate_reconstruction,
    validate_tensor_bundle,
)


def test_outlier_channel_strategy_extracts_salient_channels_and_reconstructs_losslessly() -> None:
    rng = np.random.default_rng(5)
    values = 0.05 * rng.standard_normal((32, 8))
    values[:, 1] += 3.0
    values[:, 5] -= 2.5

    split = split_outlier_channels(
        values,
        strategy=OutlierChannelStrategy(
            score_metric="max_abs",
            zscore_threshold=1.0,
            max_outlier_fraction=0.25,
        ),
    )
    stats = outlier_channel_stats(values, split)

    assert split.outlier_indices.tolist() == [1, 5]
    assert np.allclose(split.reconstruct(), values)
    assert stats.outlier_count == 2
    assert stats.outlier_fraction == 0.25
    assert stats.base_reconstruction.relative_l2 > 0.1
    assert stats.restored_reconstruction.rmse < 1e-12


def test_distortion_validation_passes_for_small_noise_and_fails_for_large_error() -> None:
    rng = np.random.default_rng(7)
    original = rng.standard_normal((16, 8))

    good = original + 0.001 * rng.standard_normal(original.shape)
    good_result = validate_reconstruction(
        original,
        good,
        thresholds=DistortionThresholds(
            rmse_max=0.01,
            relative_l2_max=0.01,
            mean_cosine_min=0.999,
            max_abs_error_max=0.01,
        ),
    )
    assert good_result.passed

    bad = np.zeros_like(original)
    bad_result = validate_reconstruction(
        original,
        bad,
        thresholds=DistortionThresholds(
            rmse_max=0.5,
            relative_l2_max=0.5,
            mean_cosine_min=0.9,
        ),
    )
    assert not bad_result.passed
    assert any(failure.startswith("mean_cosine<") or failure.startswith("relative_l2>") for failure in bad_result.failures)


def test_attention_preservation_validation_detects_bad_candidate() -> None:
    query = np.array([1.0, 0.0, 0.0, 0.0])
    keys = np.eye(4)
    values = np.eye(4)
    candidate_values = np.flipud(values)

    result = validate_attention_preservation(
        query,
        keys,
        values,
        keys,
        candidate_values,
        thresholds=DistortionThresholds(
            rmse_max=0.05,
            relative_l2_max=0.05,
            mean_cosine_min=0.99,
        ),
    )

    assert not result.passed


def test_hardened_real_model_validation_passes_for_near_candidate() -> None:
    rng = np.random.default_rng(11)
    k_cache = rng.standard_normal((2, 2, 12, 8))
    v_cache = rng.standard_normal((2, 2, 12, 8))
    activations = rng.standard_normal((4, 12, 8))
    logits = rng.standard_normal((12, 32))
    domains = ("system", "system") + ("code",) * 6 + ("rag",) * 4

    reference = TensorBundle(
        k_cache=k_cache,
        v_cache=v_cache,
        activations=activations,
        domains=domains,
        logits=logits,
    )
    candidate = TensorBundle(
        k_cache=k_cache + 0.001 * rng.standard_normal(k_cache.shape),
        v_cache=v_cache + 0.001 * rng.standard_normal(v_cache.shape),
        activations=activations + 0.001 * rng.standard_normal(activations.shape),
        domains=domains,
        logits=logits + 0.001 * rng.standard_normal(logits.shape),
    )

    report = compare_tensor_bundles(
        reference,
        candidate,
        thresholds=QualityValidationThresholds(
            logits_kl_max=0.01,
        ),
    )

    assert report.passed
    assert report.k_cache.passed
    assert report.v_cache.passed
    assert report.attention.passed
    assert report.activations is not None and report.activations.passed
    assert report.logits_kl_divergence is not None and report.logits_kl_divergence < 0.01


def test_hardened_real_model_validation_fails_for_degraded_candidate() -> None:
    rng = np.random.default_rng(13)
    k_cache = rng.standard_normal((2, 2, 10, 8))
    v_cache = rng.standard_normal((2, 2, 10, 8))
    activations = rng.standard_normal((3, 10, 8))
    logits = rng.standard_normal((10, 16))
    domains = ("system",) * 2 + ("code",) * 4 + ("rag",) * 4

    reference = TensorBundle(
        k_cache=k_cache,
        v_cache=v_cache,
        activations=activations,
        domains=domains,
        logits=logits,
    )
    candidate = TensorBundle(
        k_cache=np.zeros_like(k_cache),
        v_cache=np.zeros_like(v_cache),
        activations=np.zeros_like(activations),
        domains=domains,
        logits=np.flip(logits, axis=-1),
    )

    report = compare_tensor_bundles(reference, candidate)

    assert not report.passed
    assert report.failures


def test_validate_tensor_bundle_reports_outlier_metrics() -> None:
    rng = np.random.default_rng(17)
    k_cache = rng.standard_normal((2, 2, 8, 6))
    v_cache = rng.standard_normal((2, 2, 8, 6))
    k_cache[..., 0] += 4.0
    bundle = TensorBundle(k_cache=k_cache, v_cache=v_cache, domains=("code",) * 8)

    report = validate_tensor_bundle(bundle)

    assert "outlier_count" in report
    assert "outlier_fraction" in report
    assert report["outlier_count"] >= 1.0
