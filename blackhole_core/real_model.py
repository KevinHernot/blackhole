from __future__ import annotations

"""Real-model validation hooks for Blackhole prototype modules."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .distortion import (
    DistortionThresholds,
    DistortionValidationResult,
    kl_divergence,
    validate_attention_preservation,
    validate_reconstruction,
)
from .outlier_channels import OutlierChannelStats, outlier_channel_stats
from .stack import BlackholePrototype


@dataclass(frozen=True)
class TensorBundle:
    k_cache: np.ndarray
    v_cache: np.ndarray
    activations: np.ndarray | None = None
    domains: tuple[str, ...] | None = None
    logits: np.ndarray | None = None


@dataclass(frozen=True)
class QualityValidationThresholds:
    k_cache: DistortionThresholds = field(
        default_factory=lambda: DistortionThresholds(
            rmse_max=0.25,
            relative_l2_max=0.25,
            mean_cosine_min=0.95,
            max_abs_error_max=1.0,
        )
    )
    v_cache: DistortionThresholds = field(
        default_factory=lambda: DistortionThresholds(
            rmse_max=0.25,
            relative_l2_max=0.25,
            mean_cosine_min=0.95,
            max_abs_error_max=1.0,
        )
    )
    activations: DistortionThresholds = field(
        default_factory=lambda: DistortionThresholds(
            rmse_max=0.10,
            relative_l2_max=0.10,
            mean_cosine_min=0.98,
            max_abs_error_max=0.50,
        )
    )
    attention: DistortionThresholds = field(
        default_factory=lambda: DistortionThresholds(
            rmse_max=0.10,
            relative_l2_max=0.10,
            mean_cosine_min=0.98,
            max_abs_error_max=0.50,
        )
    )
    logits_kl_max: float = 0.05


@dataclass(frozen=True)
class QualityValidationReport:
    k_cache: DistortionValidationResult
    v_cache: DistortionValidationResult
    activations: DistortionValidationResult | None
    attention: DistortionValidationResult
    logits_kl_divergence: float | None
    merge_reduction_fraction: float
    route_reduction_fraction: float
    outlier_stats: OutlierChannelStats
    transport_residual_ratio: float | None
    passed: bool
    failures: tuple[str, ...]


def _load_bundle_field(data: object, key: str, *, dtype: object | None = None) -> np.ndarray:
    try:
        array = data[key]
    except ValueError as exc:
        raise ValueError(
            f"Tensor bundle field `{key}` uses an unsupported object array. "
            "Re-save the bundle with the current Blackhole tooling."
        ) from exc
    return np.asarray(array, dtype=dtype)


def _token_count_from_cache(cache: np.ndarray, *, field_name: str) -> int:
    if cache.ndim == 2:
        return int(cache.shape[0])
    if cache.ndim == 3:
        return int(cache.shape[1])
    if cache.ndim == 4:
        return int(cache.shape[2])
    raise ValueError(f"Unsupported {field_name} shape for token extraction: {cache.shape}")


def load_tensor_bundle(path: str | Path) -> TensorBundle:
    bundle_path = Path(path)
    with np.load(bundle_path, allow_pickle=False) as data:
        if "k_cache" not in data or "v_cache" not in data:
            raise ValueError("Tensor bundle must contain `k_cache` and `v_cache` arrays.")
        k_cache = _load_bundle_field(data, "k_cache", dtype=float)
        v_cache = _load_bundle_field(data, "v_cache", dtype=float)
        k_token_count = _token_count_from_cache(k_cache, field_name="k_cache")
        v_token_count = _token_count_from_cache(v_cache, field_name="v_cache")
        if k_token_count != v_token_count:
            raise ValueError(
                "k_cache and v_cache must describe the same token dimension: "
                f"{k_token_count} != {v_token_count}"
            )
        domains = None
        if "domains" in data:
            domains_array = _load_bundle_field(data, "domains", dtype=str)
            if domains_array.ndim != 1:
                raise ValueError(f"domains must be a 1D array, got shape {domains_array.shape}.")
            if domains_array.shape[0] != k_token_count:
                raise ValueError(
                    "domains must match token dimension implied by k_cache: "
                    f"{domains_array.shape[0]} != {k_token_count}"
                )
            domains = tuple(domains_array.tolist())
        activations = _load_bundle_field(data, "activations", dtype=float) if "activations" in data else None
        logits = _load_bundle_field(data, "logits", dtype=float) if "logits" in data else None
        return TensorBundle(
            k_cache=k_cache,
            v_cache=v_cache,
            activations=activations,
            domains=domains,
            logits=logits,
        )


def save_tensor_bundle(path: str | Path, bundle: TensorBundle) -> None:
    payload: dict[str, np.ndarray] = {
        "k_cache": bundle.k_cache,
        "v_cache": bundle.v_cache,
    }
    if bundle.activations is not None:
        payload["activations"] = bundle.activations
    if bundle.domains is not None:
        payload["domains"] = np.asarray(bundle.domains, dtype=str)
    if bundle.logits is not None:
        payload["logits"] = bundle.logits
    np.savez(Path(path), **payload)


def bundle_token_embeddings(bundle: TensorBundle) -> np.ndarray:
    cache = np.asarray(bundle.k_cache, dtype=float)
    if cache.ndim == 2:
        return cache
    if cache.ndim == 3:
        return np.mean(cache, axis=0)
    if cache.ndim == 4:
        return np.mean(cache, axis=(0, 1))
    raise ValueError(f"Unsupported k_cache shape for token embedding extraction: {cache.shape}")


def bundle_query_embedding(bundle: TensorBundle) -> np.ndarray:
    embeddings = bundle_token_embeddings(bundle)
    return embeddings[-1]


def bundle_value_embeddings(bundle: TensorBundle) -> np.ndarray:
    cache = np.asarray(bundle.v_cache, dtype=float)
    if cache.ndim == 2:
        return cache
    if cache.ndim == 3:
        return np.mean(cache, axis=0)
    if cache.ndim == 4:
        return np.mean(cache, axis=(0, 1))
    raise ValueError(f"Unsupported v_cache shape for token embedding extraction: {cache.shape}")


def validate_tensor_bundle(
    bundle: TensorBundle,
    prototype: BlackholePrototype | None = None,
    *,
    current_domain: str | None = None,
) -> dict[str, float]:
    prototype = prototype or BlackholePrototype()
    embeddings = bundle_token_embeddings(bundle)
    domains = bundle.domains or ("runtime",) * embeddings.shape[0]
    active_domain = current_domain or domains[-1]

    prepared = prototype.prepare_context(embeddings, domains)
    active = prototype.active_context(bundle_query_embedding(bundle), prepared, active_domain)
    outlier_split = prototype.split_outlier_channels(embeddings)
    outlier_stats = outlier_channel_stats(embeddings, outlier_split)

    report = {
        "original_tokens": float(embeddings.shape[0]),
        "merged_tokens": float(prepared.merged_tokens.shape[0]),
        "active_tokens": float(active.active_tokens.shape[0]),
        "merge_reduction_fraction": prepared.merge_result.reduction_fraction,
        "route_reduction_fraction": 1.0 - (active.active_tokens.shape[0] / max(prepared.merged_tokens.shape[0], 1)),
        "outlier_count": float(outlier_stats.outlier_count),
        "outlier_fraction": outlier_stats.outlier_fraction,
        "outlier_energy_fraction": outlier_stats.outlier_energy_fraction,
        "outlier_base_relative_l2": outlier_stats.base_reconstruction.relative_l2,
    }

    if bundle.activations is not None and bundle.activations.shape[0] >= 3:
        packet, stats = prototype.encode_transport(
            bundle.activations[-1],
            bundle.activations[-2],
            bundle.activations[-3],
        )
        report.update(
            {
                "transport_residual_ratio": stats.residual_ratio,
                "transport_compression_ratio": stats.compression_ratio,
                "transport_rmse": stats.reconstruction.rmse,
                "transport_bit_width": float(packet.bit_width),
            }
        )

    return report


def compare_tensor_bundles(
    reference_bundle: TensorBundle,
    candidate_bundle: TensorBundle,
    *,
    prototype: BlackholePrototype | None = None,
    thresholds: QualityValidationThresholds | None = None,
    current_domain: str | None = None,
) -> QualityValidationReport:
    prototype = prototype or BlackholePrototype()
    thresholds = thresholds or QualityValidationThresholds()

    reference_embeddings = bundle_token_embeddings(reference_bundle)
    candidate_embeddings = bundle_token_embeddings(candidate_bundle)
    reference_values = bundle_value_embeddings(reference_bundle)
    candidate_values = bundle_value_embeddings(candidate_bundle)
    domains = reference_bundle.domains or candidate_bundle.domains or ("runtime",) * reference_embeddings.shape[0]
    active_domain = current_domain or domains[-1]

    prepared = prototype.prepare_context(reference_embeddings, domains)
    active = prototype.active_context(bundle_query_embedding(reference_bundle), prepared, active_domain)
    outlier_stats = outlier_channel_stats(reference_embeddings, prototype.split_outlier_channels(reference_embeddings))

    k_cache_validation = validate_reconstruction(
        reference_bundle.k_cache,
        candidate_bundle.k_cache,
        thresholds=thresholds.k_cache,
    )
    v_cache_validation = validate_reconstruction(
        reference_bundle.v_cache,
        candidate_bundle.v_cache,
        thresholds=thresholds.v_cache,
    )
    attention_validation = validate_attention_preservation(
        bundle_query_embedding(reference_bundle),
        reference_embeddings,
        reference_values,
        candidate_embeddings,
        candidate_values,
        thresholds=thresholds.attention,
    )

    activation_validation: DistortionValidationResult | None = None
    transport_residual_ratio: float | None = None
    if reference_bundle.activations is not None and candidate_bundle.activations is not None:
        activation_validation = validate_reconstruction(
            reference_bundle.activations,
            candidate_bundle.activations,
            thresholds=thresholds.activations,
        )
    if reference_bundle.activations is not None and reference_bundle.activations.shape[0] >= 3:
        _, transport_stats = prototype.encode_transport(
            reference_bundle.activations[-1],
            reference_bundle.activations[-2],
            reference_bundle.activations[-3],
        )
        transport_residual_ratio = transport_stats.residual_ratio

    logits_kl_divergence: float | None = None
    failures = list(k_cache_validation.failures) + list(v_cache_validation.failures) + list(attention_validation.failures)
    if activation_validation is not None:
        failures.extend(activation_validation.failures)
    if reference_bundle.logits is not None and candidate_bundle.logits is not None:
        logits_kl_divergence = kl_divergence(reference_bundle.logits, candidate_bundle.logits)
        if logits_kl_divergence > thresholds.logits_kl_max:
            failures.append(f"logits_kl>{thresholds.logits_kl_max}")

    return QualityValidationReport(
        k_cache=k_cache_validation,
        v_cache=v_cache_validation,
        activations=activation_validation,
        attention=attention_validation,
        logits_kl_divergence=logits_kl_divergence,
        merge_reduction_fraction=prepared.merge_result.reduction_fraction,
        route_reduction_fraction=1.0 - (active.active_tokens.shape[0] / max(prepared.merged_tokens.shape[0], 1)),
        outlier_stats=outlier_stats,
        transport_residual_ratio=transport_residual_ratio,
        passed=not failures,
        failures=tuple(failures),
    )


def extract_transformers_kv(
    model_name: str,
    prompt: str,
    *,
    device: str = "cpu",
    trust_remote_code: bool = True,
    include_hidden_states: bool = True,
    include_logits: bool = True,
) -> TensorBundle:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "extract_transformers_kv requires optional dependencies: torch and transformers."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        device_map=device,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=include_hidden_states,
        )

    k_tensors = []
    v_tensors = []
    for layer_cache in outputs.past_key_values:
        key, value = tuple(layer_cache)
        k_tensors.append(key.squeeze(0).detach().cpu().numpy())
        v_tensors.append(value.squeeze(0).detach().cpu().numpy())

    activations = None
    if include_hidden_states and getattr(outputs, "hidden_states", None):
        hidden_states = outputs.hidden_states[-3:]
        activations = np.stack([state.squeeze(0).detach().cpu().numpy() for state in hidden_states])

    logits = None
    if include_logits and getattr(outputs, "logits", None) is not None:
        logits = outputs.logits.squeeze(0).detach().cpu().numpy()

    return TensorBundle(
        k_cache=np.stack(k_tensors),
        v_cache=np.stack(v_tensors),
        activations=activations,
        logits=logits,
    )


__all__ = [
    "QualityValidationReport",
    "QualityValidationThresholds",
    "TensorBundle",
    "bundle_value_embeddings",
    "compare_tensor_bundles",
    "bundle_query_embedding",
    "bundle_token_embeddings",
    "extract_transformers_kv",
    "load_tensor_bundle",
    "save_tensor_bundle",
    "validate_tensor_bundle",
]
