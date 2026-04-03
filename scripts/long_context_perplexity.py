#!/usr/bin/env python3
"""Long-context perplexity proof of concept for Blackhole.

This script is intentionally a *proof of concept*: it shows how each Blackhole
variant changes the 32K quality frontier on top of the standard ``q8_0``
baseline using the shared q8_0-primary ladder.
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
from _scenario_model import long_context_perplexity_metrics

SECTION_KEY = "long_context_perplexity"
CONFIGURATIONS = script_configurations("long_context_perplexity.py")

NOTES = {
    F16: "Full-precision dense reference with the lowest modeled long-context loss.",
    Q8_0: "Quantization keeps most of the fp16 long-context behavior intact.",
    Q8_0_SEMANTIC_PVS: "Semantic routing reduces irrelevant long-context competition before decode.",
    Q8_0_PORTAL_ATTENTION: "Portal windows keep only the active semantic room hot, which improves context stability.",
    Q8_0_PREDICTIVE_TRANSPORT: "Predictive hand-offs help state continuity, but do less for context curation than routing or portals.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural layouts preserve the best single-pillar long-context quality.",
    Q8_0_TOKEN_MERGING: "Merging helps throughput, but fused spans slightly increase long-context loss.",
    BLACKHOLE_ALL: "The full stack combines routing, locality, and procedural recovery into the strongest 32K quality result.",
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
                "Error: measured-quality artifacts must include a q8_0 row to anchor relative perplexity."
            )
        missing_perplexity = [
            configuration
            for configuration, metrics in metrics_by_configuration.items()
            if metrics.long_context_perplexity is None or metrics.stability_fraction is None
        ]
        if missing_perplexity:
            raise SystemExit(
                "Error: long-context artifact mode requires measured long-context perplexity and "
                f"stability fields for: {', '.join(missing_perplexity)}."
            )
        measurement_note = (
            "Measurement model: "
            f"{artifact.metadata.evidence_tier.value} artifact "
            f"from {artifact.metadata.corpus_name}, not runtime speed benchmarking."
        )
        scenario_text = (
            "Scenario: a measured long-context quality sweep sourced from a measured-quality artifact. "
            "Lower perplexity is better; stability is the short-context perplexity divided by the "
            "long-context perplexity for the same configuration."
        )
        source_note = (
            f"Artifact source: {Path(args.artifact)}\n"
            f"Reference configuration: {artifact.reference_configuration}"
        )
        baseline_ppl = metrics_by_configuration[Q8_0].long_context_perplexity
    else:
        rendered_configurations = CONFIGURATIONS
        metrics_by_configuration = {
            configuration: long_context_perplexity_metrics(configuration)
            for configuration in rendered_configurations
        }
        measurement_note = None
        scenario_text = (
            "Scenario: a 32K context quality sweep where lower perplexity is better. The stability "
            "column estimates how well each configuration preserves its short-context behavior once "
            "the prompt reaches long-context scale."
        )
        source_note = "Source: deterministic scenario-model proxy."
        baseline_ppl = metrics_by_configuration[Q8_0].ppl_proxy_32k

    print("=" * 72)
    print(" Long-Context Perplexity Proof of Concept ")
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
                (
                    f"{metrics_by_configuration[configuration].long_context_perplexity:.2f}"
                    if args.source == "artifact"
                    else f"{metrics_by_configuration[configuration].ppl_proxy_32k:.2f}"
                ),
                (
                    f"{baseline_ppl / metrics_by_configuration[configuration].long_context_perplexity:.2f}x"
                    if args.source == "artifact"
                    else f"{baseline_ppl / metrics_by_configuration[configuration].ppl_proxy_32k:.2f}x"
                ),
                (
                    f"{metrics_by_configuration[configuration].stability_fraction * 100:.1f}%"
                    if args.source == "artifact"
                    else f"{metrics_by_configuration[configuration].stability_fraction * 100:.1f}%"
                ),
                NOTES[configuration],
            )
            for configuration in rendered_configurations
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "32K perplexity" if args.source == "artifact" else "32K PPL proxy",
                "vs q8_0",
                "Stability",
                "Why it improves q8_0",
            ),
            rows,
        )
    )
    print()
    print(
        "Takeaway: lower perplexity is better. Procedural Weights is the strongest single-pillar "
        "quality stabilizer at 32K, while the full Blackhole stack closes most of the long-context "
        "gap back toward fp16 without giving up the compression story."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
