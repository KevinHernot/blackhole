from __future__ import annotations

"""Shared numeric mechanics for the Blackhole proof-of-concept suite.

Every proof-of-concept script should derive its proxy numbers from this module
so the narrative stays consistent: the same Blackhole configuration should imply
the same core mechanics everywhere, even when different scripts focus on
different sections.

These values are deterministic scenario-model proxies, not measured runtime
execution from a live transformer implementation.

Developer proxy map:

| Pillar | Primary proxy families | Main helpers |
| --- | --- | --- |
| Semantic PVS | Active-window pruning, dense skip leverage, MoE block/value culling, retrieval frontier collapse | `_prefill_active_ratio()`, `dense_skip_rate()`, `_attention_active_ratio()`, `_retrieval_scan_ratio()`, `retrieval_probability()` |
| Portal Attention | Active-window shrinkage, decode locality, retrieval band collapse, smaller transport payloads | `_prefill_active_ratio()`, `dense_skip_rate()`, `_attention_active_ratio()`, `_retrieval_scan_ratio()`, `retrieval_probability()`, `transport_volume_gb()` |
| Predictive Transport | Layer handoff efficiency, decode spillover gains, retrieval stability bonuses | `prefill_metrics()`, `dense_skip_rate()`, `moe_decode_metrics()`, `retrieval_metrics()`, `retrieval_probability()`, `transport_volume_gb()` |
| Procedural Weights | Compression-quality frontier, decode stability, retrieval stability, transport reconstruction quality | `compression_ratio()`, `quality_proxy()`, `prefill_metrics()`, `dense_skip_rate()`, `moe_decode_metrics()`, `retrieval_metrics()`, `retrieval_probability()`, `transport_volume_gb()` |
| Token Merging | Sequence-dimension reduction via greedy clustering, transport payload shrinkage, retrieval trade-offs | `_prefill_active_ratio()`, `_attention_active_ratio()`, `transport_volume_gb()`, `retrieval_probability()` |
"""

from dataclasses import dataclass
import math
from typing import Sequence

from .comparison_profiles import (
    BLACKHOLE_ALL,
    F16,
    Q8_0,
    Q8_0_PORTAL_ATTENTION,
    Q8_0_PREDICTIVE_TRANSPORT,
    Q8_0_PROCEDURAL_WEIGHTS,
    Q8_0_SEMANTIC_PVS,
    Q8_0_TOKEN_MERGING,
    canonicalize_configuration,
)

MEASUREMENT_MODE = "deterministic-scenario-proxy"
MEASURED_RUNTIME = False

PREFILL_TOTAL_TOKENS = 6_144
MOE_TOTAL_TOKENS = 16_384
RETRIEVAL_TOTAL_BLOCKS = 128
RETRIEVAL_NEEDLES = 3
TRANSPORT_FP16_VOLUME_GB = 2.00
DEFAULT_DENSE_CONTEXTS = (512, 2_048, 4_096, 8_192, 16_384, 32_768)
TOP_OF_TREE_CONTEXTS = (512, 2_048, 8_192)

# Calibration notes:
#
# This module is intentionally a proof-of-concept proxy layer, not a measured
# runtime model. The coefficients below therefore fall into two explicit
# buckets:
#
# 1. Anchor-derived ratios
#    These are computed from the target token/block counts surfaced in the
#    README tables so every script agrees on the same headline scenarios.
# 2. Heuristic gain / penalty terms
#    These are named and grouped here so future retunes can explain why a
#    narrative proxy changed instead of leaving bare floating-point literals
#    scattered throughout the file.

# Prefill anchors from the portal scenario table.
PREFILL_ACTIVE_TOKENS_SEMANTIC_ONLY = 4_096
PREFILL_ACTIVE_TOKENS_PORTAL_ONLY = 2_240
PREFILL_ACTIVE_TOKENS_ALL = 1_536
PREFILL_ACTIVE_RATIO_SEMANTIC_ONLY = PREFILL_ACTIVE_TOKENS_SEMANTIC_ONLY / PREFILL_TOTAL_TOKENS
PREFILL_ACTIVE_RATIO_PORTAL_ONLY = PREFILL_ACTIVE_TOKENS_PORTAL_ONLY / PREFILL_TOTAL_TOKENS
PREFILL_ACTIVE_RATIO_ALL = PREFILL_ACTIVE_TOKENS_ALL / PREFILL_TOTAL_TOKENS
PREFILL_SEMANTIC_RATIO_DELTA = 1.0 - PREFILL_ACTIVE_RATIO_SEMANTIC_ONLY
PREFILL_PORTAL_RATIO_DELTA = 1.0 - PREFILL_ACTIVE_RATIO_PORTAL_ONLY
PREFILL_OVERLAP_RECOVERY = PREFILL_ACTIVE_RATIO_ALL - (
    1.0 - PREFILL_SEMANTIC_RATIO_DELTA - PREFILL_PORTAL_RATIO_DELTA
)
PREFILL_ACTIVE_RATIO_MIN = 0.20
PREFILL_SPEED_COMPRESSION_BASE = 0.018
PREFILL_SPEED_COMPRESSION_SLOPE = 0.035
PREFILL_SPEED_SEMANTIC_BONUS = 0.32
PREFILL_SPEED_PORTAL_BONUS = 1.55
PREFILL_SPEED_PREDICTIVE_BONUS = 0.07
PREFILL_SPEED_PROCEDURAL_BONUS = 0.05
PREFILL_SPEED_TOKEN_MERGING_BONUS = 0.25
PREFILL_SPEED_SEMANTIC_PORTAL_SYNERGY = 0.72
PREFILL_SPEED_PORTAL_PREDICTIVE_SYNERGY = 0.10
PREFILL_SPEED_PROCEDURAL_ROUTING_SYNERGY = 0.05

# MoE decode anchors from the Semantic PVS routing table.
MOE_ACTIVE_TOKENS_SEMANTIC_ONLY = 4_096
MOE_ACTIVE_TOKENS_PORTAL_ONLY = 6_144
MOE_ACTIVE_TOKENS_ALL = 2_048
MOE_ACTIVE_RATIO_SEMANTIC_ONLY = MOE_ACTIVE_TOKENS_SEMANTIC_ONLY / MOE_TOTAL_TOKENS
MOE_ACTIVE_RATIO_PORTAL_ONLY = MOE_ACTIVE_TOKENS_PORTAL_ONLY / MOE_TOTAL_TOKENS
MOE_ACTIVE_RATIO_ALL = MOE_ACTIVE_TOKENS_ALL / MOE_TOTAL_TOKENS
ATTENTION_SEMANTIC_RATIO_DELTA = 1.0 - MOE_ACTIVE_RATIO_SEMANTIC_ONLY
ATTENTION_PORTAL_RATIO_DELTA = 1.0 - MOE_ACTIVE_RATIO_PORTAL_ONLY
ATTENTION_OVERLAP_RECOVERY = MOE_ACTIVE_RATIO_ALL - (
    1.0 - ATTENTION_SEMANTIC_RATIO_DELTA - ATTENTION_PORTAL_RATIO_DELTA
)
ATTENTION_ACTIVE_RATIO_MIN = MOE_ACTIVE_RATIO_ALL
MOE_VALUE_SKIP_OFFSET = 0.18
MOE_ACTIVE_COST_WEIGHT = 0.45
MOE_VALUE_COST_WEIGHT = 0.55
F16_MOE_BASE_FACTOR = 0.85
Q8_0_MOE_BASE_FACTOR = 0.93
MOE_COST_EXPONENT = 0.38
MOE_PREDICTIVE_BONUS = 0.08
MOE_PROCEDURAL_BONUS = 0.05
MOE_COMPRESSION_BONUS = 0.02

# Dense skip-rate proxy calibration.
Q8_0_SKIP_BASELINE_CONTEXT = 512
Q8_0_SKIP_BASE_RATE = 0.09
Q8_0_SKIP_GROWTH_PER_OCTAVE = 0.085
Q8_0_SKIP_RATE_CAP = 0.56
F16_DENSE_SKIP_FACTOR = 0.85
DENSE_SKIP_SEMANTIC_BONUS = 0.07
DENSE_SKIP_PORTAL_BONUS = 0.05
DENSE_SKIP_PREDICTIVE_BONUS = 0.03
DENSE_SKIP_PROCEDURAL_BONUS = 0.04
DENSE_SKIP_SEMANTIC_PORTAL_SYNERGY = 0.06
DENSE_SKIP_RATE_MIN = 0.01
DENSE_SKIP_RATE_MAX = 0.95

# Retrieval anchors from the NIAH routing table.
RETRIEVAL_BLOCKS_SEMANTIC_ONLY = 8
RETRIEVAL_BLOCKS_PORTAL_ONLY = 32
RETRIEVAL_BLOCKS_ALL = 6
RETRIEVAL_SCAN_RATIO_SEMANTIC_ONLY = RETRIEVAL_BLOCKS_SEMANTIC_ONLY / RETRIEVAL_TOTAL_BLOCKS
RETRIEVAL_SCAN_RATIO_PORTAL_ONLY = RETRIEVAL_BLOCKS_PORTAL_ONLY / RETRIEVAL_TOTAL_BLOCKS
RETRIEVAL_SCAN_RATIO_ALL = RETRIEVAL_BLOCKS_ALL / RETRIEVAL_TOTAL_BLOCKS
RETRIEVAL_SCAN_SEMANTIC_DELTA = 1.0 - RETRIEVAL_SCAN_RATIO_SEMANTIC_ONLY
RETRIEVAL_SCAN_PORTAL_DELTA = 1.0 - RETRIEVAL_SCAN_RATIO_PORTAL_ONLY
RETRIEVAL_SCAN_OVERLAP_RECOVERY = RETRIEVAL_SCAN_RATIO_ALL - (
    1.0 - RETRIEVAL_SCAN_SEMANTIC_DELTA - RETRIEVAL_SCAN_PORTAL_DELTA
)
RETRIEVAL_SCAN_RATIO_MIN = RETRIEVAL_SCAN_RATIO_ALL
F16_RETRIEVAL_PROXY = 0.93
Q8_0_RETRIEVAL_PROXY = 0.97
F16_BASE_RETRIEVAL_ACCURACY = 0.92
Q8_0_BASE_RETRIEVAL_ACCURACY = 0.93
RETRIEVAL_PROXY_SEMANTIC_BONUS = 0.42
RETRIEVAL_PROXY_PORTAL_BONUS = 0.14
RETRIEVAL_PROXY_PREDICTIVE_BONUS = 0.08
RETRIEVAL_PROXY_PROCEDURAL_BONUS = 0.06
RETRIEVAL_PROXY_SEMANTIC_PORTAL_SYNERGY = 0.09
RETRIEVAL_PROXY_PREDICTIVE_PROCEDURAL_SYNERGY = 0.04
RETRIEVAL_BASE_ACCURACY_BASE = 0.94
RETRIEVAL_BASE_ACCURACY_SEMANTIC_BONUS = 0.02
RETRIEVAL_BASE_ACCURACY_PORTAL_BONUS = 0.014
RETRIEVAL_BASE_ACCURACY_PREDICTIVE_BONUS = 0.01
RETRIEVAL_BASE_ACCURACY_PROCEDURAL_BONUS = 0.012
RETRIEVAL_BASE_ACCURACY_SEMANTIC_PORTAL_SYNERGY = 0.008
RETRIEVAL_BASE_ACCURACY_PREDICTIVE_PROCEDURAL_SYNERGY = 0.004
RETRIEVAL_BASE_ACCURACY_MAX = 0.985

# Retrieval probability penalty calibration.
DEPTH_PENALTY_SCALE = 0.12
CONTEXT_PENALTY_PER_OCTAVE = 0.045
SINGLE_MODE_PENALTY = 0.00
MULTI_KEY_MODE_PENALTY = 0.07
MULTI_VALUE_MODE_PENALTY = 0.05
PORTAL_CONTEXT_RESILIENCE_BONUS = 0.12
PROCEDURAL_CONTEXT_RESILIENCE_BONUS = 0.10
PREDICTIVE_MODE_RESILIENCE_BONUS = 0.08
PROCEDURAL_MODE_RESILIENCE_BONUS = 0.10
DISTRACTOR_PENALTY = 0.012
DISTRACTOR_SEMANTIC_RESILIENCE = 0.35
DISTRACTOR_PORTAL_RESILIENCE = 0.15
ADDITIONAL_VALUE_PENALTY = 0.015
VALUE_SEMANTIC_RESILIENCE = 0.20
VALUE_PREDICTIVE_RESILIENCE = 0.10
VALUE_PROCEDURAL_RESILIENCE = 0.20
QUALITY_RETRIEVAL_BONUS = 0.01
RETRIEVAL_PROBABILITY_MIN = 0.05
RETRIEVAL_PROBABILITY_MAX = 0.995

# Transport proxy calibration.
Q8_LAYOUT_FACTOR = 0.96
BLACKHOLE_LAYOUT_FACTOR_BASE = 0.92
PROCEDURAL_LAYOUT_DISCOUNT = 0.04
TRANSPORT_COMPRESSION_EXPONENT = 0.10
TRANSPORT_SEMANTIC_REDUCTION = 0.04
TRANSPORT_PORTAL_REDUCTION = 0.09
TRANSPORT_PREDICTIVE_REDUCTION = 0.16
TRANSPORT_PROCEDURAL_REDUCTION = 0.06
TRANSPORT_SEMANTIC_PORTAL_SYNERGY = 0.03
TRANSPORT_PREDICTIVE_PROCEDURAL_SYNERGY = 0.08
TRANSPORT_PORTAL_PREDICTIVE_SYNERGY = 0.04

# Token Merging (Greedy Meshing) calibration.
TOKEN_MERGING_SEQUENCE_REDUCTION = 0.35
TOKEN_MERGING_TRANSPORT_REDUCTION_FACTOR = 0.35
TOKEN_MERGING_SINGLE_RETRIEVAL_PENALTY = 0.035
TOKEN_MERGING_MULTI_VALUE_RETRIEVAL_BONUS = 0.02

# Quality-proxy calibration.
QUALITY_KLD_BASE_WEIGHT = 0.40
QUALITY_KLD_COMPRESSION_WEIGHT = 0.10
QUALITY_TOP_P_KLD_SCALE = 2.0
QUALITY_TOP_P_MIN = 0.94
LONG_CONTEXT_BASE_PPL = 7.60
LONG_CONTEXT_PPL_PER_QUALITY_POINT = 40.0
LONG_CONTEXT_STABILITY_QUALITY_SCALE = 1.60
LONG_CONTEXT_STABILITY_SEMANTIC_BONUS = 0.012
LONG_CONTEXT_STABILITY_PORTAL_BONUS = 0.020
LONG_CONTEXT_STABILITY_PREDICTIVE_BONUS = 0.014
LONG_CONTEXT_STABILITY_PROCEDURAL_BONUS = 0.018
LONG_CONTEXT_STABILITY_TOKEN_MERGING_PENALTY = 0.010
LONG_CONTEXT_STABILITY_SEMANTIC_PORTAL_SYNERGY = 0.008
LONG_CONTEXT_STABILITY_PREDICTIVE_PROCEDURAL_SYNERGY = 0.006
LONG_CONTEXT_STABILITY_MIN = 0.90
LONG_CONTEXT_PPL_INSTABILITY_SCALE = 2.25
COMPRESSION_QUALITY_MSE_FROM_KLD_SCALE = 0.60
COMPRESSION_QUALITY_FRONTIER_EXPONENT = 2.0


@dataclass(frozen=True)
class ConfigurationMechanics:
    label: str
    compression_ratio: float
    quality_cosine: float
    semantic_routing: float = 0.0
    portal_attention: float = 0.0
    predictive_transport: float = 0.0
    procedural_weights: float = 0.0
    token_merging: float = 0.0


@dataclass(frozen=True)
class PrefillMetrics:
    active_kv_tokens: int
    kv_reduction_fraction: float
    prefill_speed_proxy: float


@dataclass(frozen=True)
class MoeDecodeMetrics:
    active_key_tokens: int
    values_processed: int
    compute_reduction_fraction: float
    decode_speed_proxy: float


@dataclass(frozen=True)
class RetrievalMetrics:
    blocks_scanned: int
    needles_routed: int
    retrieval_proxy: float
    base_accuracy: float


@dataclass(frozen=True)
class TransportMetrics:
    transported_volume_gb: float
    reduction_vs_f16: float
    speed_proxy_vs_baseline: float


@dataclass(frozen=True)
class KLDivergenceMetrics:
    mean_kld: float
    same_top_p_fraction: float


@dataclass(frozen=True)
class LongContextPerplexityMetrics:
    ppl_proxy_32k: float
    stability_fraction: float


@dataclass(frozen=True)
class CompressionQualityMetrics:
    compression_ratio: float
    quality_cosine: float
    mse_proxy: float
    frontier_vs_baseline: float


CONFIGURATION_MECHANICS: dict[str, ConfigurationMechanics] = {
    F16: ConfigurationMechanics(
        label=F16,
        compression_ratio=1.0,
        quality_cosine=1.000,
    ),
    Q8_0: ConfigurationMechanics(
        label=Q8_0,
        compression_ratio=1.9,
        quality_cosine=0.994,
    ),
    Q8_0_SEMANTIC_PVS: ConfigurationMechanics(
        label=Q8_0_SEMANTIC_PVS,
        compression_ratio=1.9,
        quality_cosine=0.996,
        semantic_routing=1.0,
    ),
    Q8_0_PORTAL_ATTENTION: ConfigurationMechanics(
        label=Q8_0_PORTAL_ATTENTION,
        compression_ratio=1.9,
        quality_cosine=0.996,
        portal_attention=1.0,
    ),
    Q8_0_PREDICTIVE_TRANSPORT: ConfigurationMechanics(
        label=Q8_0_PREDICTIVE_TRANSPORT,
        compression_ratio=1.9,
        quality_cosine=0.995,
        predictive_transport=1.0,
    ),
    Q8_0_PROCEDURAL_WEIGHTS: ConfigurationMechanics(
        label=Q8_0_PROCEDURAL_WEIGHTS,
        compression_ratio=2.3,
        quality_cosine=0.998,
        procedural_weights=1.0,
    ),
    Q8_0_TOKEN_MERGING: ConfigurationMechanics(
        label=Q8_0_TOKEN_MERGING,
        compression_ratio=2.2,
        quality_cosine=0.992,
        token_merging=1.0,
    ),
    BLACKHOLE_ALL: ConfigurationMechanics(
        label=BLACKHOLE_ALL,
        compression_ratio=2.8,
        quality_cosine=0.999,
        semantic_routing=1.0,
        portal_attention=1.0,
        predictive_transport=1.0,
        procedural_weights=1.0,
        token_merging=1.0,
    ),
}


def mechanics(configuration: str) -> ConfigurationMechanics:
    return CONFIGURATION_MECHANICS[canonicalize_configuration(configuration)]


def compression_ratio(configuration: str) -> float:
    return mechanics(configuration).compression_ratio


def quality_proxy(configuration: str) -> float:
    return mechanics(configuration).quality_cosine


def _round_to(value: float, quantum: int) -> int:
    return max(quantum, int(round(value / quantum) * quantum))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _prefill_active_ratio(configuration: str) -> float:
    cfg = mechanics(configuration)
    ratio = (
        1.0
        - PREFILL_SEMANTIC_RATIO_DELTA * cfg.semantic_routing
        - PREFILL_PORTAL_RATIO_DELTA * cfg.portal_attention
        + PREFILL_OVERLAP_RECOVERY * cfg.semantic_routing * cfg.portal_attention
    )
    ratio *= 1.0 - TOKEN_MERGING_SEQUENCE_REDUCTION * cfg.token_merging
    return _clamp(ratio, PREFILL_ACTIVE_RATIO_MIN, 1.0)


def prefill_metrics(configuration: str) -> PrefillMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    active_kv_tokens = _round_to(PREFILL_TOTAL_TOKENS * _prefill_active_ratio(canonical), 64)
    kv_reduction_fraction = 1.0 - (active_kv_tokens / PREFILL_TOTAL_TOKENS)

    if canonical == F16:
        prefill_speed_proxy = 0.95
    elif canonical == Q8_0:
        prefill_speed_proxy = 1.00
    else:
        compression_gain = PREFILL_SPEED_COMPRESSION_BASE + PREFILL_SPEED_COMPRESSION_SLOPE * (
            cfg.compression_ratio - compression_ratio(Q8_0)
        )
        prefill_speed_proxy = (
            1.0
            + compression_gain
            + PREFILL_SPEED_SEMANTIC_BONUS * cfg.semantic_routing
            + PREFILL_SPEED_PORTAL_BONUS * cfg.portal_attention
            + PREFILL_SPEED_PREDICTIVE_BONUS * cfg.predictive_transport
            + PREFILL_SPEED_PROCEDURAL_BONUS * cfg.procedural_weights
            + PREFILL_SPEED_TOKEN_MERGING_BONUS * cfg.token_merging
            + PREFILL_SPEED_SEMANTIC_PORTAL_SYNERGY * cfg.semantic_routing * cfg.portal_attention
            + PREFILL_SPEED_PORTAL_PREDICTIVE_SYNERGY
            * cfg.portal_attention
            * cfg.predictive_transport
            + PREFILL_SPEED_PROCEDURAL_ROUTING_SYNERGY
            * cfg.procedural_weights
            * (cfg.semantic_routing + cfg.portal_attention)
        )

    return PrefillMetrics(
        active_kv_tokens=active_kv_tokens,
        kv_reduction_fraction=kv_reduction_fraction,
        prefill_speed_proxy=round(prefill_speed_proxy, 2),
    )


def _attention_active_ratio(configuration: str) -> float:
    cfg = mechanics(configuration)
    ratio = (
        1.0
        - ATTENTION_SEMANTIC_RATIO_DELTA * cfg.semantic_routing
        - ATTENTION_PORTAL_RATIO_DELTA * cfg.portal_attention
        + ATTENTION_OVERLAP_RECOVERY * cfg.semantic_routing * cfg.portal_attention
    )
    ratio *= 1.0 - TOKEN_MERGING_SEQUENCE_REDUCTION * cfg.token_merging
    return _clamp(ratio, ATTENTION_ACTIVE_RATIO_MIN, 1.0)


def q8_0_base_skip_rate(context_length: int) -> float:
    scaled = max(context_length, Q8_0_SKIP_BASELINE_CONTEXT) / Q8_0_SKIP_BASELINE_CONTEXT
    return min(
        Q8_0_SKIP_RATE_CAP,
        Q8_0_SKIP_BASE_RATE + Q8_0_SKIP_GROWTH_PER_OCTAVE * math.log2(scaled),
    )


def dense_skip_rate(configuration: str, context_length: int) -> float:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    base = q8_0_base_skip_rate(context_length)

    if canonical == F16:
        rate = base * F16_DENSE_SKIP_FACTOR
    elif canonical == Q8_0:
        rate = base
    else:
        rate = (
            base
            + DENSE_SKIP_SEMANTIC_BONUS * cfg.semantic_routing
            + DENSE_SKIP_PORTAL_BONUS * cfg.portal_attention
            + DENSE_SKIP_PREDICTIVE_BONUS * cfg.predictive_transport
            + DENSE_SKIP_PROCEDURAL_BONUS * cfg.procedural_weights
            + DENSE_SKIP_SEMANTIC_PORTAL_SYNERGY * cfg.semantic_routing * cfg.portal_attention
        )

    return _clamp(rate, DENSE_SKIP_RATE_MIN, DENSE_SKIP_RATE_MAX)


def average_dense_skip_rate(
    configuration: str,
    contexts: Sequence[int] = DEFAULT_DENSE_CONTEXTS,
) -> float:
    return sum(dense_skip_rate(configuration, context) for context in contexts) / len(contexts)


def dense_decode_proxy(
    configuration: str,
    contexts: Sequence[int] = DEFAULT_DENSE_CONTEXTS,
) -> float:
    baseline_average = average_dense_skip_rate(Q8_0, contexts)
    return average_dense_skip_rate(configuration, contexts) / baseline_average


def moe_decode_metrics(configuration: str, *, context_length: int = 8_192) -> MoeDecodeMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    active_key_tokens = _round_to(MOE_TOTAL_TOKENS * _attention_active_ratio(canonical), 256)
    moe_value_skip_rate = max(0.0, dense_skip_rate(canonical, context_length) - MOE_VALUE_SKIP_OFFSET)
    values_processed = max(1, int(round(active_key_tokens * (1.0 - moe_value_skip_rate))))
    compute_reduction_fraction = 1.0 - (values_processed / MOE_TOTAL_TOKENS)

    baseline_cost = _moe_cost_components(Q8_0, context_length)[0]
    this_cost = _moe_cost_components(canonical, context_length)[0]

    if canonical == F16:
        base_factor = F16_MOE_BASE_FACTOR
    else:
        base_factor = 1.0

    decode_speed_proxy = (
        base_factor
        * (baseline_cost / this_cost) ** MOE_COST_EXPONENT
        * (
            1.0
            + MOE_PREDICTIVE_BONUS * cfg.predictive_transport
            + MOE_PROCEDURAL_BONUS * cfg.procedural_weights
            + MOE_COMPRESSION_BONUS
            * max(0.0, cfg.compression_ratio - compression_ratio(Q8_0))
        )
    )

    return MoeDecodeMetrics(
        active_key_tokens=active_key_tokens,
        values_processed=values_processed,
        compute_reduction_fraction=compute_reduction_fraction,
        decode_speed_proxy=round(decode_speed_proxy, 2),
    )


def _moe_cost_components(configuration: str, context_length: int) -> tuple[float, int, int]:
    active_key_tokens = _round_to(MOE_TOTAL_TOKENS * _attention_active_ratio(configuration), 256)
    moe_value_skip_rate = max(0.0, dense_skip_rate(configuration, context_length) - MOE_VALUE_SKIP_OFFSET)
    values_processed = max(1, int(round(active_key_tokens * (1.0 - moe_value_skip_rate))))
    active_ratio = active_key_tokens / MOE_TOTAL_TOKENS
    value_ratio = values_processed / MOE_TOTAL_TOKENS
    effective_cost = MOE_ACTIVE_COST_WEIGHT * active_ratio + MOE_VALUE_COST_WEIGHT * value_ratio
    return effective_cost, active_key_tokens, values_processed


def _retrieval_scan_ratio(configuration: str) -> float:
    cfg = mechanics(configuration)
    ratio = (
        1.0
        - RETRIEVAL_SCAN_SEMANTIC_DELTA * cfg.semantic_routing
        - RETRIEVAL_SCAN_PORTAL_DELTA * cfg.portal_attention
        + RETRIEVAL_SCAN_OVERLAP_RECOVERY * cfg.semantic_routing * cfg.portal_attention
    )
    return _clamp(ratio, RETRIEVAL_SCAN_RATIO_MIN, 1.0)


def retrieval_metrics(configuration: str) -> RetrievalMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    blocks_scanned = max(1, int(round(RETRIEVAL_TOTAL_BLOCKS * _retrieval_scan_ratio(canonical))))

    if canonical == F16:
        retrieval_proxy = F16_RETRIEVAL_PROXY
        base_accuracy = F16_BASE_RETRIEVAL_ACCURACY
    elif canonical == Q8_0:
        retrieval_proxy = 1.0
        base_accuracy = Q8_0_BASE_RETRIEVAL_ACCURACY
    else:
        retrieval_proxy = (
            1.0
            + RETRIEVAL_PROXY_SEMANTIC_BONUS * cfg.semantic_routing
            + RETRIEVAL_PROXY_PORTAL_BONUS * cfg.portal_attention
            + RETRIEVAL_PROXY_PREDICTIVE_BONUS * cfg.predictive_transport
            + RETRIEVAL_PROXY_PROCEDURAL_BONUS * cfg.procedural_weights
            + RETRIEVAL_PROXY_SEMANTIC_PORTAL_SYNERGY * cfg.semantic_routing * cfg.portal_attention
            + RETRIEVAL_PROXY_PREDICTIVE_PROCEDURAL_SYNERGY
            * cfg.predictive_transport
            * cfg.procedural_weights
        )
        base_accuracy = min(
            RETRIEVAL_BASE_ACCURACY_MAX,
            RETRIEVAL_BASE_ACCURACY_BASE
            + RETRIEVAL_BASE_ACCURACY_SEMANTIC_BONUS * cfg.semantic_routing
            + RETRIEVAL_BASE_ACCURACY_PORTAL_BONUS * cfg.portal_attention
            + RETRIEVAL_BASE_ACCURACY_PREDICTIVE_BONUS * cfg.predictive_transport
            + RETRIEVAL_BASE_ACCURACY_PROCEDURAL_BONUS * cfg.procedural_weights
            + RETRIEVAL_BASE_ACCURACY_SEMANTIC_PORTAL_SYNERGY
            * cfg.semantic_routing
            * cfg.portal_attention
            + RETRIEVAL_BASE_ACCURACY_PREDICTIVE_PROCEDURAL_SYNERGY
            * cfg.predictive_transport
            * cfg.procedural_weights,
        )

    return RetrievalMetrics(
        blocks_scanned=blocks_scanned,
        needles_routed=RETRIEVAL_NEEDLES,
        retrieval_proxy=round(retrieval_proxy, 2),
        base_accuracy=base_accuracy,
    )


def _depth_penalty(depth_pct: float) -> float:
    return (abs(depth_pct - 50.0) / 50.0) * DEPTH_PENALTY_SCALE


def _context_penalty(context_length: int) -> float:
    if context_length <= 4_096:
        return 0.0
    return CONTEXT_PENALTY_PER_OCTAVE * math.log2(context_length / 4_096.0)


def retrieval_probability(
    configuration: str,
    context_length: int,
    depth_pct: float,
    mode: str,
    *,
    num_distractors: int = 0,
    value_count: int = 1,
) -> float:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    retrieval = retrieval_metrics(canonical)
    resilience = retrieval.retrieval_proxy
    mode_penalty = {
        "single": SINGLE_MODE_PENALTY,
        "multi-key": MULTI_KEY_MODE_PENALTY,
        "multi-value": MULTI_VALUE_MODE_PENALTY,
    }[mode]

    probability = retrieval.base_accuracy
    probability -= _depth_penalty(depth_pct) / resilience
    probability -= _context_penalty(context_length) / (
        resilience
        * (
            1.0
            + PORTAL_CONTEXT_RESILIENCE_BONUS * cfg.portal_attention
            + PROCEDURAL_CONTEXT_RESILIENCE_BONUS * cfg.procedural_weights
        )
    )
    probability -= mode_penalty / (
        resilience
        * (
            1.0
            + PREDICTIVE_MODE_RESILIENCE_BONUS * cfg.predictive_transport
            + PROCEDURAL_MODE_RESILIENCE_BONUS * cfg.procedural_weights
        )
    )
    probability -= num_distractors * DISTRACTOR_PENALTY / (
        1.0
        + DISTRACTOR_SEMANTIC_RESILIENCE * cfg.semantic_routing
        + DISTRACTOR_PORTAL_RESILIENCE * cfg.portal_attention
    )
    probability -= max(0, value_count - 1) * ADDITIONAL_VALUE_PENALTY / (
        1.0
        + VALUE_SEMANTIC_RESILIENCE * cfg.semantic_routing
        + VALUE_PREDICTIVE_RESILIENCE * cfg.predictive_transport
        + VALUE_PROCEDURAL_RESILIENCE * cfg.procedural_weights
    )
    probability += QUALITY_RETRIEVAL_BONUS * max(0.0, cfg.quality_cosine - quality_proxy(Q8_0))
    if mode == "single":
        probability -= TOKEN_MERGING_SINGLE_RETRIEVAL_PENALTY * cfg.token_merging
    elif mode == "multi-value":
        probability += TOKEN_MERGING_MULTI_VALUE_RETRIEVAL_BONUS * cfg.token_merging

    return _clamp(probability, RETRIEVAL_PROBABILITY_MIN, RETRIEVAL_PROBABILITY_MAX)


def transport_volume_gb(configuration: str) -> float:
    canonical = canonicalize_configuration(configuration)
    if canonical == F16:
        return TRANSPORT_FP16_VOLUME_GB

    cfg = mechanics(canonical)
    if canonical == Q8_0:
        layout_factor = Q8_LAYOUT_FACTOR
    else:
        layout_factor = BLACKHOLE_LAYOUT_FACTOR_BASE - PROCEDURAL_LAYOUT_DISCOUNT * cfg.procedural_weights

    volume = TRANSPORT_FP16_VOLUME_GB * layout_factor / (
        cfg.compression_ratio ** TRANSPORT_COMPRESSION_EXPONENT
    )
    volume *= (
        1.0
        - TRANSPORT_SEMANTIC_REDUCTION * cfg.semantic_routing
        - TRANSPORT_PORTAL_REDUCTION * cfg.portal_attention
        - TRANSPORT_PREDICTIVE_REDUCTION * cfg.predictive_transport
        - TRANSPORT_PROCEDURAL_REDUCTION * cfg.procedural_weights
    )
    volume *= 1.0 - TRANSPORT_SEMANTIC_PORTAL_SYNERGY * cfg.semantic_routing * cfg.portal_attention
    volume *= (
        1.0
        - TRANSPORT_PREDICTIVE_PROCEDURAL_SYNERGY
        * cfg.predictive_transport
        * cfg.procedural_weights
    )
    volume *= 1.0 - TRANSPORT_PORTAL_PREDICTIVE_SYNERGY * cfg.portal_attention * cfg.predictive_transport
    volume *= 1.0 - TOKEN_MERGING_TRANSPORT_REDUCTION_FACTOR * cfg.token_merging
    return round(volume, 2)


def transport_metrics(configuration: str) -> TransportMetrics:
    volume = transport_volume_gb(configuration)
    baseline_volume = transport_volume_gb(Q8_0)
    return TransportMetrics(
        transported_volume_gb=volume,
        reduction_vs_f16=1.0 - (volume / TRANSPORT_FP16_VOLUME_GB),
        speed_proxy_vs_baseline=baseline_volume / volume,
    )


def kl_divergence_metrics(configuration: str) -> KLDivergenceMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    quality_gap = max(0.0, 1.0 - cfg.quality_cosine)
    mean_kld = quality_gap * (
        QUALITY_KLD_BASE_WEIGHT
        + QUALITY_KLD_COMPRESSION_WEIGHT * max(0.0, cfg.compression_ratio - 1.0)
    )
    same_top_p_fraction = _clamp(
        1.0 - QUALITY_TOP_P_KLD_SCALE * mean_kld,
        QUALITY_TOP_P_MIN,
        1.0,
    )
    return KLDivergenceMetrics(
        mean_kld=round(mean_kld, 4),
        same_top_p_fraction=round(same_top_p_fraction, 4),
    )


def long_context_perplexity_metrics(configuration: str) -> LongContextPerplexityMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    quality_gap = max(0.0, 1.0 - cfg.quality_cosine)
    short_context_ppl = LONG_CONTEXT_BASE_PPL + LONG_CONTEXT_PPL_PER_QUALITY_POINT * quality_gap
    stability = (
        1.0
        - LONG_CONTEXT_STABILITY_QUALITY_SCALE * quality_gap
        + LONG_CONTEXT_STABILITY_SEMANTIC_BONUS * cfg.semantic_routing
        + LONG_CONTEXT_STABILITY_PORTAL_BONUS * cfg.portal_attention
        + LONG_CONTEXT_STABILITY_PREDICTIVE_BONUS * cfg.predictive_transport
        + LONG_CONTEXT_STABILITY_PROCEDURAL_BONUS * cfg.procedural_weights
        - LONG_CONTEXT_STABILITY_TOKEN_MERGING_PENALTY * cfg.token_merging
        + LONG_CONTEXT_STABILITY_SEMANTIC_PORTAL_SYNERGY
        * cfg.semantic_routing
        * cfg.portal_attention
        + LONG_CONTEXT_STABILITY_PREDICTIVE_PROCEDURAL_SYNERGY
        * cfg.predictive_transport
        * cfg.procedural_weights
    )
    stability = _clamp(stability, LONG_CONTEXT_STABILITY_MIN, 1.0)
    ppl_proxy_32k = short_context_ppl * (
        1.0 + LONG_CONTEXT_PPL_INSTABILITY_SCALE * (1.0 - stability)
    )
    return LongContextPerplexityMetrics(
        ppl_proxy_32k=round(ppl_proxy_32k, 2),
        stability_fraction=round(stability, 4),
    )


def compression_quality_metrics(configuration: str) -> CompressionQualityMetrics:
    canonical = canonicalize_configuration(configuration)
    cfg = mechanics(canonical)
    drift = kl_divergence_metrics(canonical)
    frontier_vs_baseline = (
        cfg.compression_ratio / compression_ratio(Q8_0)
    ) * (cfg.quality_cosine / quality_proxy(Q8_0)) ** COMPRESSION_QUALITY_FRONTIER_EXPONENT
    return CompressionQualityMetrics(
        compression_ratio=cfg.compression_ratio,
        quality_cosine=cfg.quality_cosine,
        mse_proxy=round(drift.mean_kld * COMPRESSION_QUALITY_MSE_FROM_KLD_SCALE, 4),
        frontier_vs_baseline=round(frontier_vs_baseline, 2),
    )


def top_of_tree_summary(
    configuration: str,
    *,
    moe_context_length: int = 8_192,
) -> tuple[str, str, str, str, str, str]:
    prefill = prefill_metrics(configuration)
    moe = moe_decode_metrics(configuration, context_length=moe_context_length)
    retrieval = retrieval_metrics(configuration)
    transport = transport_metrics(configuration)
    return (
        f"{compression_ratio(configuration):.1f}x",
        f"{prefill.prefill_speed_proxy:.2f}x",
        f"{moe.decode_speed_proxy:.2f}x",
        f"{retrieval.retrieval_proxy:.2f}x",
        f"{transport.speed_proxy_vs_baseline:.2f}x",
        f"{quality_proxy(configuration):.3f}",
    )


__all__ = [
    "BLACKHOLE_ALL",
    "CompressionQualityMetrics",
    "ConfigurationMechanics",
    "KLDivergenceMetrics",
    "LongContextPerplexityMetrics",
    "DEFAULT_DENSE_CONTEXTS",
    "DENSE_SKIP_RATE_MAX",
    "DENSE_SKIP_RATE_MIN",
    "F16",
    "MEASUREMENT_MODE",
    "MEASURED_RUNTIME",
    "MOE_TOTAL_TOKENS",
    "MoeDecodeMetrics",
    "PREFILL_TOTAL_TOKENS",
    "PrefillMetrics",
    "Q8_0",
    "RETRIEVAL_PROBABILITY_MAX",
    "RETRIEVAL_PROBABILITY_MIN",
    "RETRIEVAL_NEEDLES",
    "RETRIEVAL_TOTAL_BLOCKS",
    "RetrievalMetrics",
    "TOP_OF_TREE_CONTEXTS",
    "TRANSPORT_FP16_VOLUME_GB",
    "Q8_0_PORTAL_ATTENTION",
    "Q8_0_PREDICTIVE_TRANSPORT",
    "Q8_0_PROCEDURAL_WEIGHTS",
    "Q8_0_SEMANTIC_PVS",
    "Q8_0_TOKEN_MERGING",
    "TransportMetrics",
    "average_dense_skip_rate",
    "compression_quality_metrics",
    "compression_ratio",
    "dense_decode_proxy",
    "dense_skip_rate",
    "kl_divergence_metrics",
    "long_context_perplexity_metrics",
    "mechanics",
    "moe_decode_metrics",
    "prefill_metrics",
    "quality_proxy",
    "retrieval_metrics",
    "retrieval_probability",
    "top_of_tree_summary",
    "q8_0_base_skip_rate",
    "transport_metrics",
    "transport_volume_gb",
]
