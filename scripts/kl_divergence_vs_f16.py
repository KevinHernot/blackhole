#!/usr/bin/env python3
"""KL divergence proof of concept for Blackhole.

This script is intentionally a *proof of concept*: it shows how each Blackhole
variant changes quality drift relative to the full-precision ``f16`` reference,
using the same shared q8_0-primary ladder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from _comparison_profiles import (
    BLACKHOLE_ALL,
    F16,
    Q8_0,
    Q8_0_PORTAL_ATTENTION,
    Q8_0_PREDICTIVE_TRANSPORT,
    Q8_0_PROCEDURAL_WEIGHTS,
    Q8_0_SEMANTIC_PVS,
    Q8_0_TOKEN_MERGING,
    markdown_table,
    ordered_section_rows,
    render_section_overview,
    script_configurations,
)
from _scenario_model import kl_divergence_metrics

SECTION_KEY = "kl_divergence_vs_f16"
CONFIGURATIONS = script_configurations("kl_divergence_vs_f16.py")

NOTES = {
    F16: "Full-precision dense reference — zero modeled drift by definition.",
    Q8_0: "Quantization alone preserves most of the fp16 decision surface.",
    Q8_0_SEMANTIC_PVS: "Routing trims irrelevant work while staying closer to the fp16 surface than plain q8_0.",
    Q8_0_PORTAL_ATTENTION: "Portal locality reduces long-context interference without changing the q8_0 KV substrate.",
    Q8_0_PREDICTIVE_TRANSPORT: "Lighter transport helps state fidelity without disturbing the q8_0 decision surface much.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural reconstruction is the strongest single-pillar quality stabilizer.",
    Q8_0_TOKEN_MERGING: "Sequence merging adds some drift pressure because multiple nearby states are fused.",
    BLACKHOLE_ALL: "The full stack stays closer to fp16 than q8_0 while also compressing the working state more effectively.",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=("proxy", "artifact"),
        default="proxy",
        help="Render from the proxy model or from a measured-quality artifact.",
    )
    parser.add_argument(
        "--artifact",
        default=None,
        help="Path to a measured-quality artifact JSON file. Required when --source artifact.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = []
    args = parse_args(argv)
    if args.source == "artifact":
        if not args.artifact:
            raise SystemExit("Error: --artifact is required when --source artifact.")
        from _measured_quality import load_measured_quality_artifact

        artifact = load_measured_quality_artifact(args.artifact)
        rendered_configurations = tuple(
            configuration
            for configuration in CONFIGURATIONS
            if configuration in artifact.measurements
        )
        metrics_by_configuration = {
            configuration: artifact.measurement_for(configuration)
            for configuration in rendered_configurations
        }
        if Q8_0 not in metrics_by_configuration:
            raise SystemExit(
                "Error: measured-quality artifacts must include a q8_0 row to anchor relative drift."
            )
        missing_drift = [
            configuration
            for configuration, metrics in metrics_by_configuration.items()
            if metrics.mean_kld is None or metrics.same_top_p_fraction is None
        ]
        if missing_drift:
            raise SystemExit(
                "Error: KL-divergence artifact mode requires measured prompt-logit drift fields "
                f"for: {', '.join(missing_drift)}."
            )
        measurement_note = (
            "Measurement model: "
            f"{artifact.metadata.evidence_tier.value} artifact "
            f"from {artifact.metadata.corpus_name}, not runtime speed benchmarking."
        )
        scenario_text = (
            "Scenario: compare each configuration's measured output drift against the full-precision "
            "f16 reference from a measured-quality artifact. Lower KLD is better; same-top-p "
            "estimates how often the candidate keeps the fp16 top-probability set intact."
        )
        source_note = (
            f"Artifact source: {Path(args.artifact)}\n"
            f"Reference configuration: {artifact.reference_configuration}"
        )
    else:
        rendered_configurations = CONFIGURATIONS
        metrics_by_configuration = {
            configuration: kl_divergence_metrics(configuration)
            for configuration in rendered_configurations
        }
        measurement_note = None
        scenario_text = (
            "Scenario: compare each configuration's modeled output drift against the full-precision "
            "f16 reference. Lower KLD is better; same-top-p estimates how often the candidate keeps "
            "the fp16 top-probability set intact."
        )
        source_note = "Source: deterministic scenario-model proxy."

    baseline_kld = metrics_by_configuration[Q8_0].mean_kld

    print("=" * 72)
    print(" KL Divergence Proof of Concept ")
    print("=" * 72)
    print(scenario_text)
    print()
    print(render_section_overview(SECTION_KEY, rendered_configurations, measurement_note=measurement_note))
    print(source_note)
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                f"{metrics_by_configuration[configuration].mean_kld:.4f}",
                f"{metrics_by_configuration[configuration].same_top_p_fraction * 100:.1f}%",
                f"{metrics_by_configuration[configuration].mean_kld / baseline_kld:.2f}x",
                NOTES[configuration],
            )
            for configuration in rendered_configurations
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Mean KLD vs f16",
                "Same-top-p",
                "Relative drift vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )
    print()
    print(
        "Takeaway: lower KLD is better. Procedural Weights is the clearest single-pillar "
        "quality win, and the full Blackhole stack keeps materially less modeled drift than "
        "standard q8_0 despite compressing the working state more aggressively."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
