from __future__ import annotations

"""Shared Blackhole proof-of-concept catalog.

The Blackhole Python scripts are intentionally *proofs of concept*, not
runtime-backed benchmarks. Every script therefore reports the same
q8_0-primary ladder and explains how each variant improves on the
standard ``q8_0`` baseline.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

F16 = "f16"
Q8_0 = "q8_0"
Q8_0_SEMANTIC_PVS = "q8_0 + Semantic PVS"
Q8_0_PORTAL_ATTENTION = "q8_0 + Portal Attention"
Q8_0_PREDICTIVE_TRANSPORT = "q8_0 + Predictive Transport"
Q8_0_PROCEDURAL_WEIGHTS = "q8_0 + Procedural Weights"
Q8_0_TOKEN_MERGING = "q8_0 + Token Merging"
BLACKHOLE_ALL = "blackhole (q8_0 + all 5)"
BLACKHOLE_BASELINE = Q8_0

SCENARIO_MODEL_NOTE = (
    "Measurement model: deterministic scenario-model proxies, not measured runtime execution."
)

ALLOWED_CONFIGURATION_LABELS = (
    F16,
    Q8_0,
    Q8_0_SEMANTIC_PVS,
    Q8_0_PORTAL_ATTENTION,
    Q8_0_PREDICTIVE_TRANSPORT,
    Q8_0_PROCEDURAL_WEIGHTS,
    Q8_0_TOKEN_MERGING,
    BLACKHOLE_ALL,
)


@dataclass(frozen=True)
class ConfigurationProfile:
    label: str
    family: str
    description: str
    pillars: tuple[str, ...] = ()


@dataclass(frozen=True)
class SectionProfile:
    key: str
    title: str
    description: str
    configurations: tuple[str, ...]
    primary_metrics: tuple[str, ...]


CONFIGURATION_PROFILES: dict[str, ConfigurationProfile] = {
    F16: ConfigurationProfile(
        label=F16,
        family="baseline",
        description="Full-precision dense reference.",
    ),
    Q8_0: ConfigurationProfile(
        label=Q8_0,
        family="baseline",
        description="8-bit KV cache baseline before Blackhole-specific ideas enter the picture.",
    ),
    Q8_0_SEMANTIC_PVS: ConfigurationProfile(
        label=Q8_0_SEMANTIC_PVS,
        family="blackhole-pillar",
        description="Q8_0 plus Semantic PVS routing for macro-culling and retrieval focus.",
        pillars=("Semantic PVS",),
    ),
    Q8_0_PORTAL_ATTENTION: ConfigurationProfile(
        label=Q8_0_PORTAL_ATTENTION,
        family="blackhole-pillar",
        description="Q8_0 plus Portal Attention for domain-local context activation.",
        pillars=("Portal Attention",),
    ),
    Q8_0_PREDICTIVE_TRANSPORT: ConfigurationProfile(
        label=Q8_0_PREDICTIVE_TRANSPORT,
        family="blackhole-pillar",
        description="Q8_0 plus Predictive Transport for lighter layer-to-layer hand-offs.",
        pillars=("Predictive Transport",),
    ),
    Q8_0_PROCEDURAL_WEIGHTS: ConfigurationProfile(
        label=Q8_0_PROCEDURAL_WEIGHTS,
        family="blackhole-pillar",
        description="Q8_0 plus Procedural Weights for a better compression-quality frontier.",
        pillars=("Procedural Weights",),
    ),
    Q8_0_TOKEN_MERGING: ConfigurationProfile(
        label=Q8_0_TOKEN_MERGING,
        family="blackhole-pillar",
        description="Q8_0 plus Token Merging (Greedy Meshing) for sequence-dimension reduction.",
        pillars=("Token Merging",),
    ),
    BLACKHOLE_ALL: ConfigurationProfile(
        label=BLACKHOLE_ALL,
        family="blackhole-full",
        description="Full Blackhole stack: q8_0 plus all five additive pillars.",
        pillars=(
            "Semantic PVS",
            "Portal Attention",
            "Predictive Transport",
            "Procedural Weights",
            "Token Merging",
        ),
    ),
}


RESULT_SECTION_ORDER = (
    "top_of_tree_results",
    "prefill_context_scaling",
    "decode_speed_moe",
    "niah_retrieval",
    "kl_divergence_vs_f16",
    "decode_speed_dense",
    "long_context_perplexity",
    "compression_quality",
    "speed_optimization_journey",
    "sequence_compression_tome",
)


def _full_ladder_section(
    key: str,
    title: str,
    description: str,
    primary_metrics: tuple[str, ...],
) -> SectionProfile:
    return SectionProfile(
        key=key,
        title=title,
        description=description,
        configurations=ALLOWED_CONFIGURATION_LABELS,
        primary_metrics=primary_metrics,
    )


SECTION_PROFILES: dict[str, SectionProfile] = {
    "top_of_tree_results": _full_ladder_section(
        "top_of_tree_results",
        "Top-of-Tree Results",
        "Full proof-of-concept ladder comparing every Blackhole configuration against standard q8_0.",
        ("compression", "speed proxy", "quality proxy", "note"),
    ),
    "prefill_context_scaling": _full_ladder_section(
        "prefill_context_scaling",
        "Prefill Context Scaling (Verified 2K-32K)",
        "Portal and routing ideas are presented as direct improvements on top of the standard q8_0 prefill story.",
        ("active KV", "prefill proxy", "vs q8_0"),
    ),
    "decode_speed_moe": _full_ladder_section(
        "decode_speed_moe",
        "Decode Speed — MoE",
        "Every configuration is shown on the same MoE decode ladder so the gains over q8_0 are explicit.",
        ("active keys", "values processed", "decode proxy", "vs q8_0"),
    ),
    "niah_retrieval": _full_ladder_section(
        "niah_retrieval",
        "NIAH Retrieval",
        "Retrieval proofs of concept compare the full Blackhole ladder instead of hiding behind runtime subsets.",
        ("blocks scanned", "needles routed", "vs q8_0", "note"),
    ),
    "kl_divergence_vs_f16": _full_ladder_section(
        "kl_divergence_vs_f16",
        "KL Divergence vs f16",
        "Quality-proxy section measuring how far each variant drifts from the full-precision reference.",
        ("mean KLD", "same-top-p", "vs q8_0"),
    ),
    "decode_speed_dense": _full_ladder_section(
        "decode_speed_dense",
        "Decode Speed — Dense",
        "Dense-model proof of concept showing how each Blackhole pillar changes the q8_0 decode budget.",
        ("skip leverage", "decode proxy", "vs q8_0"),
    ),
    "long_context_perplexity": _full_ladder_section(
        "long_context_perplexity",
        "Long-Context Perplexity (Primary Quality Metric)",
        "Primary quality section for the full Blackhole ladder relative to the q8_0 baseline.",
        ("32K PPL proxy", "vs q8_0", "stability"),
    ),
    "compression_quality": _full_ladder_section(
        "compression_quality",
        "Compression Quality (Python Prototype)",
        "Compression proof of concept comparing the full q8_0-primary ladder on the same scale.",
        ("compression", "cosine", "mse", "vs q8_0"),
    ),
    "speed_optimization_journey": _full_ladder_section(
        "speed_optimization_journey",
        "Speed Optimization Journey",
        "Incremental view of how each Blackhole pillar improves the standard q8_0 architecture.",
        ("speed proxy", "vs q8_0", "incremental note"),
    ),
    "sequence_compression_tome": _full_ladder_section(
        "sequence_compression_tome",
        "Sequence Compression — Greedy Meshing",
        "Token Merging proof of concept mapping 3D greedy meshing to sequence reduction.",
        ("original length", "merged length", "sequence reduction", "speed proxy", "vs q8_0"),
    ),
}


SCRIPT_SECTION_MAP: dict[str, tuple[str, ...]] = {
    "compression_quality.py": ("compression_quality",),
    "kl_divergence_vs_f16.py": ("kl_divergence_vs_f16",),
    "long_context_perplexity.py": ("long_context_perplexity",),
    "measure_skip_rate.py": ("decode_speed_dense",),
    "niah_pvs_routing.py": ("niah_retrieval",),
    "niah_test.py": ("niah_retrieval",),
    "portal_attention.py": ("prefill_context_scaling",),
    "predictive_transport.py": ("speed_optimization_journey",),
    "semantic_pvs_routing.py": ("decode_speed_moe", "decode_speed_dense"),
    "token_merging_poc.py": ("sequence_compression_tome",),
    "unified_poc.py": RESULT_SECTION_ORDER,
}

SCRIPT_CONFIGURATION_MAP: dict[str, tuple[str, ...]] = {
    script_name: ALLOWED_CONFIGURATION_LABELS
    for script_name in SCRIPT_SECTION_MAP
}


_CONFIGURATION_ALIASES = {
    "blackhole": BLACKHOLE_ALL,
    "blackhole4": BLACKHOLE_ALL,
    "q8_0+semantic pvs": Q8_0_SEMANTIC_PVS,
    "q8_0+portal attention": Q8_0_PORTAL_ATTENTION,
    "q8_0+predictive transport": Q8_0_PREDICTIVE_TRANSPORT,
    "q8_0+procedural weights": Q8_0_PROCEDURAL_WEIGHTS,
    "q8_0+token merging": Q8_0_TOKEN_MERGING,
    "blackhole (q8_0+all 5)": BLACKHOLE_ALL,
    "blackhole5": BLACKHOLE_ALL,
}

_SECTION_ALIASES: dict[str, str] = {}
for key, profile in SECTION_PROFILES.items():
    _SECTION_ALIASES[key.casefold()] = key
    _SECTION_ALIASES[key.replace("_", " ").casefold()] = key
    _SECTION_ALIASES[profile.title.casefold()] = key


def canonicalize_configuration(label: str) -> str:
    candidate = " ".join(label.strip().split())
    if not candidate:
        raise ValueError("Configuration label cannot be empty.")

    alias_key = candidate.casefold()
    if alias_key in _CONFIGURATION_ALIASES:
        return _CONFIGURATION_ALIASES[alias_key]

    for allowed in ALLOWED_CONFIGURATION_LABELS:
        if allowed.casefold() == alias_key:
            return allowed

    allowed_text = ", ".join(ALLOWED_CONFIGURATION_LABELS)
    raise ValueError(
        f"Unsupported Blackhole configuration: {label!r}. Allowed labels: {allowed_text}"
    )


def validate_configurations(labels: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for label in labels:
        canonical = canonicalize_configuration(label)
        if canonical not in seen:
            ordered.append(canonical)
            seen.add(canonical)
    return tuple(ordered)


def configuration_profile(label: str) -> ConfigurationProfile:
    return CONFIGURATION_PROFILES[canonicalize_configuration(label)]


def section_profile(name: str) -> SectionProfile:
    key = _SECTION_ALIASES.get(name.strip().casefold())
    if key is None:
        known = ", ".join(profile.title for profile in SECTION_PROFILES.values())
        raise KeyError(f"Unknown Blackhole result section: {name!r}. Known sections: {known}")
    return SECTION_PROFILES[key]


def section_configurations(name: str) -> tuple[str, ...]:
    return section_profile(name).configurations


def script_sections(script_name: str) -> tuple[str, ...]:
    return SCRIPT_SECTION_MAP.get(script_name, ())


def script_configurations(script_name: str) -> tuple[str, ...]:
    return SCRIPT_CONFIGURATION_MAP.get(script_name, ALLOWED_CONFIGURATION_LABELS)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header_row = "| " + " | ".join(str(header) for header in headers) + " |"
    separator_row = "| " + " | ".join("---" for _ in headers) + " |"
    body_rows = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header_row, separator_row, *body_rows])


def ordered_section_rows(
    section_key: str,
    values_by_configuration: Mapping[str, Sequence[object]],
) -> list[list[str]]:
    profile = section_profile(section_key)
    canonical_map = {
        canonicalize_configuration(label): [str(value) for value in values]
        for label, values in values_by_configuration.items()
    }
    rows: list[list[str]] = []
    for label in profile.configurations:
        values = canonical_map.get(label)
        if values is not None:
            rows.append([label, *values])
    return rows


def render_section_overview(
    section_key: str,
    produced_configurations: Sequence[str] | None = None,
    *,
    measurement_note: str | None = None,
) -> str:
    profile = section_profile(section_key)
    lines = [
        f"### {profile.title}",
        profile.description,
        f"Proof-of-concept ladder: {', '.join(profile.configurations)}",
        f"Primary metrics: {', '.join(profile.primary_metrics)}",
        f"Common baseline: {BLACKHOLE_BASELINE}",
        measurement_note or SCENARIO_MODEL_NOTE,
    ]

    if produced_configurations is not None:
        produced = validate_configurations(produced_configurations)
        lines.append(f"Script coverage: {', '.join(produced)}")

    return "\n".join(lines)


__all__ = [
    "ALLOWED_CONFIGURATION_LABELS",
    "BLACKHOLE_ALL",
    "BLACKHOLE_BASELINE",
    "CONFIGURATION_PROFILES",
    "ConfigurationProfile",
    "F16",
    "Q8_0",
    "Q8_0_PORTAL_ATTENTION",
    "Q8_0_PREDICTIVE_TRANSPORT",
    "Q8_0_PROCEDURAL_WEIGHTS",
    "Q8_0_SEMANTIC_PVS",
    "Q8_0_TOKEN_MERGING",
    "RESULT_SECTION_ORDER",
    "SCENARIO_MODEL_NOTE",
    "SCRIPT_CONFIGURATION_MAP",
    "SCRIPT_SECTION_MAP",
    "SECTION_PROFILES",
    "SectionProfile",
    "canonicalize_configuration",
    "configuration_profile",
    "markdown_table",
    "ordered_section_rows",
    "render_section_overview",
    "script_configurations",
    "script_sections",
    "section_configurations",
    "section_profile",
    "validate_configurations",
]
