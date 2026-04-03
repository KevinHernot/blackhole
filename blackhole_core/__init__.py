"""Blackhole proof-of-concept package.

This package exposes the shared deterministic scenario model and configuration
catalog used by the script entry points in ``scripts/``.

It also exposes the NumPy prototype surface for the five Blackhole pillars.
Those heavier modules are loaded lazily so script-only proof-of-concept entry
points can continue to import the shared scenario model without requiring the
full algorithm stack at import time.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .comparison_profiles import (
    ALLOWED_CONFIGURATION_LABELS,
    BLACKHOLE_ALL,
    BLACKHOLE_BASELINE,
    CONFIGURATION_PROFILES,
    F16,
    Q8_0,
    Q8_0_PORTAL_ATTENTION,
    Q8_0_PREDICTIVE_TRANSPORT,
    Q8_0_PROCEDURAL_WEIGHTS,
    Q8_0_SEMANTIC_PVS,
    Q8_0_TOKEN_MERGING,
    RESULT_SECTION_ORDER,
    SCENARIO_MODEL_NOTE,
    SECTION_PROFILES,
    canonicalize_configuration,
    configuration_profile,
    markdown_table,
    ordered_section_rows,
    render_section_overview,
    script_configurations,
    script_sections,
    section_configurations,
    section_profile,
    validate_configurations,
)
from .evidence_tiers import ArtifactMetadata, EvidenceTier
from .scenario_model import (
    CompressionQualityMetrics,
    DEFAULT_DENSE_CONTEXTS,
    KLDivergenceMetrics,
    LongContextPerplexityMetrics,
    MEASUREMENT_MODE,
    MEASURED_RUNTIME,
    MOE_TOTAL_TOKENS,
    PREFILL_TOTAL_TOKENS,
    RETRIEVAL_NEEDLES,
    RETRIEVAL_TOTAL_BLOCKS,
    TOP_OF_TREE_CONTEXTS,
    TRANSPORT_FP16_VOLUME_GB,
    average_dense_skip_rate,
    compression_quality_metrics,
    compression_ratio,
    dense_decode_proxy,
    dense_skip_rate,
    kl_divergence_metrics,
    long_context_perplexity_metrics,
    mechanics,
    moe_decode_metrics,
    prefill_metrics,
    quality_proxy,
    q8_0_base_skip_rate,
    retrieval_metrics,
    retrieval_probability,
    top_of_tree_summary,
    transport_metrics,
    transport_volume_gb,
)

__version__ = "0.2.0"

_OPTIONAL_DEPENDENCY_ROOTS = {
    "accelerate",
    "numpy",
    "scipy",
    "torch",
    "transformers",
}

_LAZY_MODULE_EXPORTS: dict[str, tuple[str, ...]] = {
    "benchmarks": (
        "TimingResult",
        "benchmark_blackhole_components",
        "benchmark_operation",
    ),
    "distortion": (
        "DistortionStats",
        "DistortionThresholds",
        "DistortionValidationResult",
        "attention_output",
        "kl_divergence",
        "max_abs_error",
        "stable_softmax",
        "validate_attention_preservation",
        "validate_reconstruction",
    ),
    "metrics": (
        "ReconstructionStats",
        "cosine_similarity",
        "ensure_2d",
        "l2_normalize",
        "mean_cosine_similarity",
        "mean_squared_error",
        "pairwise_cosine_scores",
        "reconstruction_stats",
        "relative_l2_error",
        "root_mean_squared_error",
    ),
    "outlier_channels": (
        "OutlierChannelSplit",
        "OutlierChannelStats",
        "OutlierChannelStrategy",
        "outlier_channel_stats",
        "restore_outlier_channels",
        "score_outlier_channels",
        "select_outlier_channels",
        "split_outlier_channels",
    ),
    "portal_attention": (
        "PortalAttentionResult",
        "activate_portal_context",
        "gather_portal_tokens",
    ),
    "predictive_transport": (
        "PredictiveTransportCodec",
        "PredictiveTransportStats",
        "QuantizedTransportPacket",
        "predict_next_activation",
    ),
    "runtime_capture_contract": (
        "ReferenceCaptureManifest",
        "ReferenceCaptureSample",
        "RuntimeCandidateManifest",
        "RuntimeCandidateSample",
        "build_runtime_candidate_manifest_template",
        "merge_runtime_capture_manifests",
    ),
    "measured_quality": (
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
    ),
    "procedural_weights": (
        "ProceduralTileSpec",
        "ProceduralizedMatrix",
        "fit_procedural_tile",
        "procedural_matrix_stats",
        "proceduralize_matrix",
        "reconstruct_procedural_matrix",
        "reconstruct_procedural_tile",
        "tile_salience",
    ),
    "real_model": (
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
    ),
    "semantic_pvs": (
        "SemanticPVSIndex",
        "SemanticPVSResult",
        "build_semantic_pvs_index",
        "gather_active_tokens",
        "relevant_block_recall",
        "route_semantic_blocks",
    ),
    "stack": (
        "ActiveBlackholeContext",
        "BlackholeConfig",
        "BlackholePrototype",
        "PreparedBlackholeContext",
    ),
    "token_merging": (
        "MergedSpan",
        "TokenMergingResult",
        "expand_merged_tokens",
        "merge_adjacent_tokens",
    ),
}

_LAZY_ATTRS = {
    export_name: module_name
    for module_name, export_names in _LAZY_MODULE_EXPORTS.items()
    for export_name in export_names
}


def _load_lazy_module(module_name: str) -> Any:
    try:
        return import_module(f".{module_name}", __name__)
    except ModuleNotFoundError as exc:
        root_name = exc.name.split(".", 1)[0] if exc.name else ""
        if root_name not in _OPTIONAL_DEPENDENCY_ROOTS:
            raise
        raise ModuleNotFoundError(
            "The Blackhole NumPy prototype surface requires optional dependencies "
            f"for `{module_name}`. Install the project dependencies from "
            "`pyproject.toml` before using that part of `blackhole_core`."
        ) from exc


def __getattr__(name: str) -> Any:
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _load_lazy_module(module_name)
    for export_name in _LAZY_MODULE_EXPORTS[module_name]:
        globals()[export_name] = getattr(module, export_name)

    return globals()[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "ALLOWED_CONFIGURATION_LABELS",
    "ArtifactMetadata",
    "BLACKHOLE_ALL",
    "BLACKHOLE_BASELINE",
    "CompressionQualityMetrics",
    "CONFIGURATION_PROFILES",
    "DEFAULT_DENSE_CONTEXTS",
    "EvidenceTier",
    "F16",
    "KLDivergenceMetrics",
    "LongContextPerplexityMetrics",
    "MeasuredQualityArtifact",
    "MeasuredQualityMetrics",
    "RuntimeObservedCapture",
    "MEASUREMENT_MODE",
    "MEASURED_RUNTIME",
    "MOE_TOTAL_TOKENS",
    "PREFILL_TOTAL_TOKENS",
    "Q8_0",
    "Q8_0_PORTAL_ATTENTION",
    "Q8_0_PREDICTIVE_TRANSPORT",
    "Q8_0_PROCEDURAL_WEIGHTS",
    "Q8_0_SEMANTIC_PVS",
    "Q8_0_TOKEN_MERGING",
    "RESULT_SECTION_ORDER",
    "RETRIEVAL_NEEDLES",
    "RETRIEVAL_TOTAL_BLOCKS",
    "SCENARIO_MODEL_NOTE",
    "SECTION_PROFILES",
    "TOP_OF_TREE_CONTEXTS",
    "TRANSPORT_FP16_VOLUME_GB",
    "average_dense_skip_rate",
    "add_frontier_vs_baseline",
    "aggregate_measured_quality_metrics",
    "build_measured_quality_artifact_from_bundle_paths",
    "bundle_serialized_bytes",
    "canonicalize_configuration",
    "compression_quality_metrics",
    "compression_ratio",
    "configuration_profile",
    "dense_decode_proxy",
    "dense_skip_rate",
    "kl_divergence_metrics",
    "long_context_perplexity_metrics",
    "load_context_eval",
    "markdown_table",
    "load_measured_quality_artifact",
    "load_runtime_observed_capture",
    "mechanics",
    "measure_quality_metrics_from_capture_paths",
    "measure_quality_metrics",
    "measure_quality_metrics_from_bundle_paths",
    "measure_quality_metrics_from_runtime_observed_paths",
    "moe_decode_metrics",
    "ordered_section_rows",
    "perplexity_from_logits",
    "prefill_metrics",
    "q8_0_base_skip_rate",
    "quality_proxy",
    "render_section_overview",
    "retrieval_metrics",
    "retrieval_probability",
    "same_top_p_fraction",
    "save_context_eval",
    "save_measured_quality_artifact",
    "script_configurations",
    "script_sections",
    "section_configurations",
    "section_profile",
    "top_of_tree_summary",
    "transport_metrics",
    "transport_volume_gb",
    "validate_configurations",
    "ActiveBlackholeContext",
    "BlackholeConfig",
    "BlackholePrototype",
    "DistortionStats",
    "DistortionThresholds",
    "DistortionValidationResult",
    "MergedSpan",
    "OutlierChannelSplit",
    "OutlierChannelStats",
    "OutlierChannelStrategy",
    "PortalAttentionResult",
    "PredictiveTransportCodec",
    "PredictiveTransportStats",
    "PreparedBlackholeContext",
    "ProceduralTileSpec",
    "ProceduralizedMatrix",
    "QualityValidationReport",
    "QualityValidationThresholds",
    "QuantizedTransportPacket",
    "ReconstructionStats",
    "ReferenceCaptureManifest",
    "ReferenceCaptureSample",
    "SemanticPVSIndex",
    "SemanticPVSResult",
    "TensorBundle",
    "TimingResult",
    "TokenMergingResult",
    "RuntimeCandidateManifest",
    "RuntimeCandidateSample",
    "activate_portal_context",
    "attention_output",
    "benchmark_blackhole_components",
    "benchmark_operation",
    "build_runtime_candidate_manifest_template",
    "build_semantic_pvs_index",
    "bundle_value_embeddings",
    "bundle_query_embedding",
    "bundle_token_embeddings",
    "compare_tensor_bundles",
    "cosine_similarity",
    "ensure_2d",
    "expand_merged_tokens",
    "extract_transformers_kv",
    "fit_procedural_tile",
    "gather_active_tokens",
    "gather_portal_tokens",
    "kl_divergence",
    "l2_normalize",
    "load_tensor_bundle",
    "max_abs_error",
    "mean_cosine_similarity",
    "mean_squared_error",
    "merge_adjacent_tokens",
    "merge_runtime_capture_manifests",
    "outlier_channel_stats",
    "pairwise_cosine_scores",
    "predict_next_activation",
    "procedural_matrix_stats",
    "proceduralize_matrix",
    "reconstruction_stats",
    "reconstruct_procedural_matrix",
    "reconstruct_procedural_tile",
    "relative_l2_error",
    "restore_outlier_channels",
    "relevant_block_recall",
    "root_mean_squared_error",
    "route_semantic_blocks",
    "save_tensor_bundle",
    "score_outlier_channels",
    "select_outlier_channels",
    "split_outlier_channels",
    "stable_softmax",
    "tile_salience",
    "validate_attention_preservation",
    "validate_reconstruction",
    "validate_tensor_bundle",
]
