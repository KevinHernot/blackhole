from __future__ import annotations

"""Measured-quality artifact helpers for Blackhole.

This module is the bridge between proof-of-concept quality scripts and real
artifact-backed measurements. It does not require a live model runtime by
itself: the inputs can come from cached tensor bundles, deterministic fixtures,
or future real-model evaluation pipelines.
"""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .comparison_profiles import Q8_0, canonicalize_configuration
from .evidence_tiers import ArtifactMetadata

if TYPE_CHECKING:
    import numpy as np

    from .real_model import TensorBundle


_TENSOR_BUNDLE_V1 = "tensor_bundle_v1"
_RUNTIME_OBSERVED_V1 = "runtime_observed_v1"


def _optional_payload_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _optional_payload_int(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


@dataclass(frozen=True)
class MeasuredQualityMetrics:
    mean_kld: float | None = None
    same_top_p_fraction: float | None = None
    mean_cosine: float | None = None
    mse: float | None = None
    relative_l2: float | None = None
    max_abs_error: float | None = None
    serialized_bytes: int | None = None
    compression_ratio: float | None = None
    short_context_perplexity: float | None = None
    long_context_perplexity: float | None = None
    stability_fraction: float | None = None
    frontier_vs_baseline: float | None = None
    sample_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasuredQualityMetrics":
        return cls(
            mean_kld=_optional_payload_float(payload, "mean_kld"),
            same_top_p_fraction=_optional_payload_float(payload, "same_top_p_fraction"),
            mean_cosine=_optional_payload_float(payload, "mean_cosine"),
            mse=_optional_payload_float(payload, "mse"),
            relative_l2=_optional_payload_float(payload, "relative_l2"),
            max_abs_error=_optional_payload_float(payload, "max_abs_error"),
            serialized_bytes=_optional_payload_int(payload, "serialized_bytes"),
            compression_ratio=_optional_payload_float(payload, "compression_ratio"),
            short_context_perplexity=_optional_payload_float(payload, "short_context_perplexity"),
            long_context_perplexity=_optional_payload_float(payload, "long_context_perplexity"),
            stability_fraction=_optional_payload_float(payload, "stability_fraction"),
            frontier_vs_baseline=_optional_payload_float(payload, "frontier_vs_baseline"),
            sample_count=_optional_payload_int(payload, "sample_count"),
        )


@dataclass(frozen=True)
class RuntimeObservedCapture:
    token_ids: np.ndarray
    token_embeddings: np.ndarray | None = None
    logits: np.ndarray | None = None


@dataclass(frozen=True)
class MeasuredQualityArtifact:
    metadata: ArtifactMetadata
    reference_configuration: str
    reference_bytes: int
    top_p_threshold: float
    measurements: Mapping[str, MeasuredQualityMetrics]

    def measurement_for(self, configuration: str) -> MeasuredQualityMetrics:
        return self.measurements[canonicalize_configuration(configuration)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "reference_configuration": self.reference_configuration,
            "reference_bytes": self.reference_bytes,
            "top_p_threshold": self.top_p_threshold,
            "measurements": {
                configuration: metrics.to_dict()
                for configuration, metrics in self.measurements.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MeasuredQualityArtifact":
        return cls(
            metadata=ArtifactMetadata.from_dict(dict(payload["metadata"])),
            reference_configuration=canonicalize_configuration(str(payload["reference_configuration"])),
            reference_bytes=int(payload["reference_bytes"]),
            top_p_threshold=float(payload["top_p_threshold"]),
            measurements={
                canonicalize_configuration(configuration): MeasuredQualityMetrics.from_dict(metrics)
                for configuration, metrics in dict(payload["measurements"]).items()
            },
        )


def _top_p_index_set(probabilities: np.ndarray, top_p_threshold: float) -> frozenset[int]:
    import numpy as np

    if not 0.0 < top_p_threshold <= 1.0:
        raise ValueError(f"top_p_threshold must be in (0, 1], got {top_p_threshold}.")

    ordered_indices = np.argsort(probabilities)[::-1]
    cumulative = 0.0
    selected: list[int] = []
    for index in ordered_indices:
        selected.append(int(index))
        cumulative += float(probabilities[index])
        if cumulative >= top_p_threshold:
            break
    return frozenset(selected)


def same_top_p_fraction(
    reference_logits: np.ndarray | list[float],
    candidate_logits: np.ndarray | list[float],
    *,
    top_p_threshold: float = 0.90,
) -> float:
    from .distortion import stable_softmax
    from .metrics import ensure_2d

    reference_probabilities = ensure_2d(stable_softmax(reference_logits, axis=-1), name="reference_probabilities")
    candidate_probabilities = ensure_2d(stable_softmax(candidate_logits, axis=-1), name="candidate_probabilities")
    if reference_probabilities.shape != candidate_probabilities.shape:
        raise ValueError(
            "reference_logits and candidate_logits must share the same shape: "
            f"{reference_probabilities.shape} != {candidate_probabilities.shape}"
        )

    matches = 0
    for reference_row, candidate_row in zip(reference_probabilities, candidate_probabilities):
        if _top_p_index_set(reference_row, top_p_threshold) == _top_p_index_set(candidate_row, top_p_threshold):
            matches += 1
    return matches / max(reference_probabilities.shape[0], 1)


def perplexity_from_logits(
    logits: np.ndarray | list[float],
    target_token_ids: np.ndarray | list[int],
) -> float:
    import numpy as np

    from .distortion import EPSILON, stable_softmax
    from .metrics import ensure_2d

    probability_rows = ensure_2d(stable_softmax(logits, axis=-1), name="probability_rows")
    token_ids = np.asarray(target_token_ids, dtype=int)
    if token_ids.ndim != 1:
        raise ValueError(f"target_token_ids must be 1D, got shape {token_ids.shape}.")
    if token_ids.shape[0] != probability_rows.shape[0]:
        raise ValueError(
            "target_token_ids must align with logits rows: "
            f"{token_ids.shape[0]} != {probability_rows.shape[0]}"
        )
    if np.any(token_ids < 0) or np.any(token_ids >= probability_rows.shape[1]):
        raise ValueError("target_token_ids contains an index outside the logits vocabulary range.")

    selected = probability_rows[np.arange(token_ids.shape[0]), token_ids]
    negative_log_likelihood = -np.log(np.maximum(selected, EPSILON))
    return float(np.exp(np.mean(negative_log_likelihood)))


def bundle_serialized_bytes(bundle: TensorBundle) -> int:
    total = int(bundle.k_cache.nbytes + bundle.v_cache.nbytes)
    if bundle.activations is not None:
        total += int(bundle.activations.nbytes)
    if bundle.logits is not None:
        total += int(bundle.logits.nbytes)
    return total


def save_context_eval(
    path: str | Path,
    logits: np.ndarray | list[list[float]],
    targets: np.ndarray | list[int],
) -> None:
    import numpy as np

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        logits=np.asarray(logits, dtype=float),
        targets=np.asarray(targets, dtype=int),
    )


def load_context_eval(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as data:
        if "logits" not in data or "targets" not in data:
            raise ValueError("Context eval file must contain `logits` and `targets` arrays.")
        return (
            np.asarray(data["logits"], dtype=float),
            np.asarray(data["targets"], dtype=int),
        )


def load_runtime_observed_capture(
    path: str | Path,
    *,
    captured_components: Sequence[str] | None = None,
) -> RuntimeObservedCapture:
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as data:
        if "token_ids" not in data:
            raise ValueError("runtime_observed_v1 capture must contain a `token_ids` array.")
        token_ids = np.asarray(data["token_ids"], dtype=int)
        token_embeddings = (
            np.asarray(data["token_embeddings"], dtype=float)
            if "token_embeddings" in data
            else None
        )
        logits = np.asarray(data["logits"], dtype=float) if "logits" in data else None

    if token_ids.ndim != 1 or token_ids.size == 0:
        raise ValueError(f"runtime_observed_v1 `token_ids` must be a non-empty 1D array, got {token_ids.shape}.")
    if token_embeddings is not None:
        if token_embeddings.ndim != 2:
            raise ValueError(
                "runtime_observed_v1 `token_embeddings` must be a 2D array, "
                f"got {token_embeddings.shape}."
            )
        if token_embeddings.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "runtime_observed_v1 `token_embeddings` must align with `token_ids`: "
                f"{token_embeddings.shape[0]} != {token_ids.shape[0]}"
            )
    if logits is not None:
        if logits.ndim != 2:
            raise ValueError(f"runtime_observed_v1 `logits` must be a 2D array, got {logits.shape}.")
        if logits.shape[0] != token_ids.shape[0]:
            raise ValueError(
                "runtime_observed_v1 `logits` must align with `token_ids`: "
                f"{logits.shape[0]} != {token_ids.shape[0]}"
            )

    expected_components = {str(component) for component in (captured_components or ())}
    missing_components: list[str] = []
    if "token_embeddings" in expected_components and token_embeddings is None:
        missing_components.append("token_embeddings")
    if "logits" in expected_components and logits is None:
        missing_components.append("logits")
    if missing_components:
        raise ValueError(
            "runtime_observed_v1 capture is missing declared components: "
            + ", ".join(sorted(missing_components))
        )
    if token_embeddings is None and logits is None:
        raise ValueError(
            "runtime_observed_v1 capture must include at least one promotable component "
            "(`token_embeddings` or `logits`)."
        )

    return RuntimeObservedCapture(
        token_ids=token_ids,
        token_embeddings=token_embeddings,
        logits=logits,
    )


def _load_context_metric_kwargs(
    *,
    short_context_path: str | Path | None = None,
    long_context_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    kwargs: dict[str, np.ndarray] = {}
    if short_context_path is not None:
        short_logits, short_targets = load_context_eval(short_context_path)
        kwargs.update(
            {
                "short_context_logits": short_logits,
                "short_context_targets": short_targets,
            }
        )
    if long_context_path is not None:
        long_logits, long_targets = load_context_eval(long_context_path)
        kwargs.update(
            {
                "long_context_logits": long_logits,
                "long_context_targets": long_targets,
            }
        )
    return kwargs


def _context_perplexity_metrics(
    *,
    short_context_logits: np.ndarray | None = None,
    short_context_targets: np.ndarray | None = None,
    long_context_logits: np.ndarray | None = None,
    long_context_targets: np.ndarray | None = None,
) -> tuple[float | None, float | None, float | None]:
    from .distortion import EPSILON

    short_context_perplexity: float | None = None
    long_context_perplexity: float | None = None
    stability_fraction: float | None = None
    if short_context_logits is not None or short_context_targets is not None:
        if short_context_logits is None or short_context_targets is None:
            raise ValueError("short_context_logits and short_context_targets must be provided together.")
        short_context_perplexity = perplexity_from_logits(short_context_logits, short_context_targets)
    if long_context_logits is not None or long_context_targets is not None:
        if long_context_logits is None or long_context_targets is None:
            raise ValueError("long_context_logits and long_context_targets must be provided together.")
        long_context_perplexity = perplexity_from_logits(long_context_logits, long_context_targets)
    if short_context_perplexity is not None and long_context_perplexity is not None:
        stability_fraction = short_context_perplexity / max(long_context_perplexity, EPSILON)
    return short_context_perplexity, long_context_perplexity, stability_fraction


def measure_quality_metrics(
    reference_bundle: TensorBundle,
    candidate_bundle: TensorBundle,
    *,
    reference_serialized_bytes: int | None = None,
    candidate_serialized_bytes: int | None = None,
    top_p_threshold: float = 0.90,
    short_context_logits: np.ndarray | None = None,
    short_context_targets: np.ndarray | None = None,
    long_context_logits: np.ndarray | None = None,
    long_context_targets: np.ndarray | None = None,
) -> MeasuredQualityMetrics:
    import numpy as np
    from .real_model import compare_tensor_bundles

    if reference_bundle.logits is None or candidate_bundle.logits is None:
        raise ValueError("Both reference_bundle and candidate_bundle must include logits.")

    report = compare_tensor_bundles(reference_bundle, candidate_bundle)
    reconstructions = [
        report.k_cache.stats.reconstruction,
        report.v_cache.stats.reconstruction,
        report.attention.stats.reconstruction,
    ]
    max_abs_error = max(
        report.k_cache.stats.max_abs_error,
        report.v_cache.stats.max_abs_error,
        report.attention.stats.max_abs_error,
    )
    if report.activations is not None:
        reconstructions.append(report.activations.stats.reconstruction)
        max_abs_error = max(max_abs_error, report.activations.stats.max_abs_error)

    reference_bytes = reference_serialized_bytes or bundle_serialized_bytes(reference_bundle)
    candidate_bytes = candidate_serialized_bytes or bundle_serialized_bytes(candidate_bundle)
    if candidate_bytes <= 0:
        raise ValueError("candidate_serialized_bytes must be positive.")

    short_context_perplexity, long_context_perplexity, stability_fraction = _context_perplexity_metrics(
        short_context_logits=short_context_logits,
        short_context_targets=short_context_targets,
        long_context_logits=long_context_logits,
        long_context_targets=long_context_targets,
    )

    return MeasuredQualityMetrics(
        mean_kld=float(report.logits_kl_divergence if report.logits_kl_divergence is not None else 0.0),
        same_top_p_fraction=same_top_p_fraction(
            reference_bundle.logits,
            candidate_bundle.logits,
            top_p_threshold=top_p_threshold,
        ),
        mean_cosine=float(np.mean([reconstruction.mean_cosine for reconstruction in reconstructions])),
        mse=float(np.mean([reconstruction.mse for reconstruction in reconstructions])),
        relative_l2=float(np.mean([reconstruction.relative_l2 for reconstruction in reconstructions])),
        max_abs_error=float(max_abs_error),
        serialized_bytes=int(candidate_bytes),
        compression_ratio=float(reference_bytes / candidate_bytes),
        short_context_perplexity=short_context_perplexity,
        long_context_perplexity=long_context_perplexity,
        stability_fraction=stability_fraction,
        frontier_vs_baseline=None,
    )


def measure_quality_metrics_from_bundle_paths(
    reference_bundle_path: str | Path,
    candidate_bundle_path: str | Path,
    *,
    reference_serialized_bytes: int | None = None,
    candidate_serialized_bytes: int | None = None,
    top_p_threshold: float = 0.90,
    short_context_path: str | Path | None = None,
    long_context_path: str | Path | None = None,
) -> MeasuredQualityMetrics:
    from .real_model import load_tensor_bundle

    return measure_quality_metrics(
        load_tensor_bundle(reference_bundle_path),
        load_tensor_bundle(candidate_bundle_path),
        reference_serialized_bytes=reference_serialized_bytes,
        candidate_serialized_bytes=candidate_serialized_bytes,
        top_p_threshold=top_p_threshold,
        **_load_context_metric_kwargs(
            short_context_path=short_context_path,
            long_context_path=long_context_path,
        ),
    )


def measure_quality_metrics_from_runtime_observed_paths(
    reference_bundle_path: str | Path,
    candidate_bundle_path: str | Path,
    *,
    reference_serialized_bytes: int | None = None,
    candidate_serialized_bytes: int | None = None,
    candidate_captured_components: Sequence[str] | None = None,
    top_p_threshold: float = 0.90,
    short_context_path: str | Path | None = None,
    long_context_path: str | Path | None = None,
) -> MeasuredQualityMetrics:
    import numpy as np

    from .distortion import kl_divergence
    from .real_model import bundle_token_embeddings, load_tensor_bundle

    reference_bundle = load_tensor_bundle(reference_bundle_path)
    observed_capture = load_runtime_observed_capture(
        candidate_bundle_path,
        captured_components=candidate_captured_components,
    )
    candidate_bytes = (
        int(candidate_serialized_bytes)
        if candidate_serialized_bytes is not None
        else int(Path(candidate_bundle_path).stat().st_size)
    )
    if candidate_bytes <= 0:
        raise ValueError("candidate_serialized_bytes must be positive.")

    mean_kld: float | None = None
    top_p_fraction: float | None = None
    if observed_capture.logits is not None:
        if reference_bundle.logits is None:
            raise ValueError(
                "reference bundle must include logits to compare against runtime_observed_v1 prompt logits."
            )
        reference_logits = np.asarray(reference_bundle.logits, dtype=float)
        if reference_logits.shape != observed_capture.logits.shape:
            raise ValueError(
                "runtime_observed_v1 prompt logits must align with reference logits: "
                f"{observed_capture.logits.shape} != {reference_logits.shape}"
            )
        mean_kld = kl_divergence(reference_logits, observed_capture.logits)
        top_p_fraction = same_top_p_fraction(
            reference_logits,
            observed_capture.logits,
            top_p_threshold=top_p_threshold,
        )

    if observed_capture.token_embeddings is not None:
        reference_embeddings = bundle_token_embeddings(reference_bundle)
        if reference_embeddings.shape != observed_capture.token_embeddings.shape:
            raise ValueError(
                "runtime_observed_v1 token embeddings must align with reference token embeddings: "
                f"{observed_capture.token_embeddings.shape} != {reference_embeddings.shape}"
            )

    context_kwargs = _load_context_metric_kwargs(
        short_context_path=short_context_path,
        long_context_path=long_context_path,
    )
    short_context_perplexity, long_context_perplexity, stability_fraction = _context_perplexity_metrics(
        short_context_logits=context_kwargs.get("short_context_logits"),
        short_context_targets=context_kwargs.get("short_context_targets"),
        long_context_logits=context_kwargs.get("long_context_logits"),
        long_context_targets=context_kwargs.get("long_context_targets"),
    )
    if mean_kld is None and short_context_perplexity is None and long_context_perplexity is None:
        raise ValueError(
            "runtime_observed_v1 captures require prompt logits or short/long context evals "
            "to produce promotable measured-quality metrics."
        )

    return MeasuredQualityMetrics(
        mean_kld=mean_kld,
        same_top_p_fraction=top_p_fraction,
        mean_cosine=None,
        mse=None,
        relative_l2=None,
        max_abs_error=None,
        serialized_bytes=int(candidate_bytes),
        compression_ratio=None,
        short_context_perplexity=short_context_perplexity,
        long_context_perplexity=long_context_perplexity,
        stability_fraction=stability_fraction,
        frontier_vs_baseline=None,
    )


def measure_quality_metrics_from_capture_paths(
    reference_bundle_path: str | Path,
    candidate_bundle_path: str | Path,
    *,
    candidate_bundle_format: str | None = None,
    candidate_captured_components: Sequence[str] | None = None,
    reference_serialized_bytes: int | None = None,
    candidate_serialized_bytes: int | None = None,
    top_p_threshold: float = 0.90,
    short_context_path: str | Path | None = None,
    long_context_path: str | Path | None = None,
) -> MeasuredQualityMetrics:
    bundle_format = str(candidate_bundle_format) if candidate_bundle_format is not None else None
    if bundle_format in (None, _TENSOR_BUNDLE_V1):
        return measure_quality_metrics_from_bundle_paths(
            reference_bundle_path,
            candidate_bundle_path,
            reference_serialized_bytes=reference_serialized_bytes,
            candidate_serialized_bytes=candidate_serialized_bytes,
            top_p_threshold=top_p_threshold,
            short_context_path=short_context_path,
            long_context_path=long_context_path,
        )
    if bundle_format == _RUNTIME_OBSERVED_V1:
        return measure_quality_metrics_from_runtime_observed_paths(
            reference_bundle_path,
            candidate_bundle_path,
            reference_serialized_bytes=reference_serialized_bytes,
            candidate_serialized_bytes=candidate_serialized_bytes,
            candidate_captured_components=candidate_captured_components,
            top_p_threshold=top_p_threshold,
            short_context_path=short_context_path,
            long_context_path=long_context_path,
        )
    raise ValueError(
        "unsupported candidate bundle_format "
        f"{bundle_format!r}; expected {_TENSOR_BUNDLE_V1!r} or {_RUNTIME_OBSERVED_V1!r}"
    )


def build_measured_quality_artifact_from_bundle_paths(
    *,
    metadata: ArtifactMetadata,
    reference_configuration: str,
    reference_bundle_path: str | Path,
    candidate_bundle_paths: Mapping[str, str | Path],
    candidate_bundle_formats: Mapping[str, str] | None = None,
    candidate_captured_components: Mapping[str, Sequence[str]] | None = None,
    candidate_serialized_bytes: Mapping[str, int] | None = None,
    top_p_threshold: float = 0.90,
    short_context_paths: Mapping[str, str | Path] | None = None,
    long_context_paths: Mapping[str, str | Path] | None = None,
) -> MeasuredQualityArtifact:
    from .real_model import load_tensor_bundle

    reference_configuration = canonicalize_configuration(reference_configuration)
    reference_bytes = bundle_serialized_bytes(load_tensor_bundle(reference_bundle_path))
    byte_overrides = {
        canonicalize_configuration(configuration): int(value)
        for configuration, value in (candidate_serialized_bytes or {}).items()
    }
    bundle_formats = {
        canonicalize_configuration(configuration): str(value)
        for configuration, value in (candidate_bundle_formats or {}).items()
    }
    captured_components = {
        canonicalize_configuration(configuration): tuple(str(component) for component in values)
        for configuration, values in (candidate_captured_components or {}).items()
    }
    short_paths = {
        canonicalize_configuration(configuration): path
        for configuration, path in (short_context_paths or {}).items()
    }
    long_paths = {
        canonicalize_configuration(configuration): path
        for configuration, path in (long_context_paths or {}).items()
    }

    measurements: dict[str, MeasuredQualityMetrics] = {}
    for configuration, candidate_path in candidate_bundle_paths.items():
        canonical = canonicalize_configuration(configuration)
        measurements[canonical] = measure_quality_metrics_from_capture_paths(
            reference_bundle_path,
            candidate_path,
            candidate_bundle_format=bundle_formats.get(canonical),
            candidate_captured_components=captured_components.get(canonical),
            reference_serialized_bytes=reference_bytes,
            candidate_serialized_bytes=byte_overrides.get(canonical),
            top_p_threshold=top_p_threshold,
            short_context_path=short_paths.get(canonical),
            long_context_path=long_paths.get(canonical),
        )

    return MeasuredQualityArtifact(
        metadata=metadata,
        reference_configuration=reference_configuration,
        reference_bytes=reference_bytes,
        top_p_threshold=top_p_threshold,
        measurements=add_frontier_vs_baseline(measurements),
    )


def aggregate_measured_quality_metrics(
    measurements: list[MeasuredQualityMetrics] | tuple[MeasuredQualityMetrics, ...],
) -> MeasuredQualityMetrics:
    if not measurements:
        raise ValueError("measurements must not be empty.")

    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values))

    def _mean_optional(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return _mean(present)

    def _mean_int_optional(values: list[int | None]) -> int | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return int(round(_mean([float(value) for value in present])))

    def _max_optional(values: list[float | None]) -> float | None:
        present = [value for value in values if value is not None]
        if not present:
            return None
        return float(max(present))

    return MeasuredQualityMetrics(
        mean_kld=_mean_optional([metrics.mean_kld for metrics in measurements]),
        same_top_p_fraction=_mean_optional([metrics.same_top_p_fraction for metrics in measurements]),
        mean_cosine=_mean_optional([metrics.mean_cosine for metrics in measurements]),
        mse=_mean_optional([metrics.mse for metrics in measurements]),
        relative_l2=_mean_optional([metrics.relative_l2 for metrics in measurements]),
        max_abs_error=_max_optional([metrics.max_abs_error for metrics in measurements]),
        serialized_bytes=_mean_int_optional([metrics.serialized_bytes for metrics in measurements]),
        compression_ratio=_mean_optional([metrics.compression_ratio for metrics in measurements]),
        short_context_perplexity=_mean_optional(
            [metrics.short_context_perplexity for metrics in measurements]
        ),
        long_context_perplexity=_mean_optional(
            [metrics.long_context_perplexity for metrics in measurements]
        ),
        stability_fraction=_mean_optional([metrics.stability_fraction for metrics in measurements]),
        frontier_vs_baseline=None,
        sample_count=len(measurements),
    )


def add_frontier_vs_baseline(
    measurements: Mapping[str, MeasuredQualityMetrics],
    *,
    baseline_label: str = Q8_0,
) -> dict[str, MeasuredQualityMetrics]:
    baseline_metrics = measurements[canonicalize_configuration(baseline_label)]
    enriched: dict[str, MeasuredQualityMetrics] = {}
    for configuration, metrics in measurements.items():
        frontier: float | None = None
        if (
            baseline_metrics.compression_ratio is not None
            and baseline_metrics.mean_cosine is not None
            and baseline_metrics.compression_ratio > 0.0
            and baseline_metrics.mean_cosine != 0.0
            and metrics.compression_ratio is not None
            and metrics.mean_cosine is not None
        ):
            frontier = (
                metrics.compression_ratio / baseline_metrics.compression_ratio
            ) * (metrics.mean_cosine / baseline_metrics.mean_cosine) ** 2
        enriched[canonicalize_configuration(configuration)] = replace(
            metrics,
            frontier_vs_baseline=float(frontier) if frontier is not None else None,
        )
    return enriched

def save_measured_quality_artifact(path: str | Path, artifact: MeasuredQualityArtifact) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact.to_dict(), indent=2) + "\n")


def load_measured_quality_artifact(path: str | Path) -> MeasuredQualityArtifact:
    payload = json.loads(Path(path).read_text())
    return MeasuredQualityArtifact.from_dict(payload)


__all__ = [
    "MeasuredQualityArtifact",
    "MeasuredQualityMetrics",
    "RuntimeObservedCapture",
    "add_frontier_vs_baseline",
    "aggregate_measured_quality_metrics",
    "build_measured_quality_artifact_from_bundle_paths",
    "bundle_serialized_bytes",
    "load_context_eval",
    "load_measured_quality_artifact",
    "load_runtime_observed_capture",
    "measure_quality_metrics_from_capture_paths",
    "measure_quality_metrics",
    "measure_quality_metrics_from_bundle_paths",
    "measure_quality_metrics_from_runtime_observed_paths",
    "perplexity_from_logits",
    "same_top_p_fraction",
    "save_context_eval",
    "save_measured_quality_artifact",
]
