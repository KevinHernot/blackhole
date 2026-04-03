#!/usr/bin/env python3
"""Compression-quality frontier proof of concept for Blackhole.

This script is intentionally a *proof of concept*: it compares the full
Blackhole ladder on the compression-quality frontier using the same shared
q8_0-primary story centered on ``q8_0``.
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
from _scenario_model import compression_quality_metrics

SECTION_KEY = "compression_quality"
CONFIGURATIONS = script_configurations("compression_quality.py")

NOTES = {
    F16: "Dense reference — best raw quality, but no compression frontier advantage.",
    Q8_0: "Strong baseline quality retention, but limited working-state compression on its own.",
    Q8_0_SEMANTIC_PVS: "Routing helps quality slightly without changing the raw compression ratio.",
    Q8_0_PORTAL_ATTENTION: "Portal locality reduces interference, nudging quality up at the same compression point.",
    Q8_0_PREDICTIVE_TRANSPORT: "Transport focuses on movement cost more than compression-quality balance.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural reconstruction is the strongest single-pillar compression-quality frontier win.",
    Q8_0_TOKEN_MERGING: "Sequence merging improves effective compression, but pays a small quality tax in isolation.",
    BLACKHOLE_ALL: "The full stack lands furthest above q8_0 on the modeled compression-quality frontier.",
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
                "Error: measured-quality artifacts must include a q8_0 row to anchor the frontier."
            )
        missing_frontier_metrics = [
            configuration
            for configuration, metrics in metrics_by_configuration.items()
            if (
                metrics.compression_ratio is None
                or metrics.mean_cosine is None
                or metrics.mse is None
                or metrics.frontier_vs_baseline is None
            )
        ]
        if missing_frontier_metrics:
            raise SystemExit(
                "Error: compression-quality artifact mode requires full tensor-bundle "
                "compression/reconstruction metrics for: "
                f"{', '.join(missing_frontier_metrics)}. "
                "Artifacts built from runtime_observed_v1 captures support KL/perplexity promotion only."
            )
        measurement_note = (
            "Measurement model: "
            f"{artifact.metadata.evidence_tier.value} artifact "
            f"from {artifact.metadata.corpus_name}, not runtime speed benchmarking."
        )
        scenario_text = (
            "Scenario: compare measured quality preservation against measured compression from a "
            "measured-quality artifact. Frontier values above 1.00x mean a better compression-quality "
            "balance than standard q8_0."
        )
        source_note = (
            f"Artifact source: {Path(args.artifact)}\n"
            f"Reference configuration: {artifact.reference_configuration}"
        )
    else:
        rendered_configurations = CONFIGURATIONS
        metrics_by_configuration = {
            configuration: compression_quality_metrics(configuration)
            for configuration in rendered_configurations
        }
        measurement_note = None
        scenario_text = (
            "Scenario: compare how much quality each configuration preserves for the amount of "
            "compression it achieves. Frontier values above 1.00x mean a better compression-quality "
            "balance than standard q8_0."
        )
        source_note = "Source: deterministic scenario-model proxy."

    print("=" * 72)
    print(" Compression Quality Proof of Concept ")
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
                    f"{metrics_by_configuration[configuration].compression_ratio:.1f}x"
                    if args.source == "proxy"
                    else f"{metrics_by_configuration[configuration].compression_ratio:.2f}x"
                ),
                (
                    f"{metrics_by_configuration[configuration].quality_cosine:.3f}"
                    if args.source == "proxy"
                    else f"{metrics_by_configuration[configuration].mean_cosine:.3f}"
                ),
                (
                    f"{metrics_by_configuration[configuration].mse_proxy:.4f}"
                    if args.source == "proxy"
                    else f"{metrics_by_configuration[configuration].mse:.4f}"
                ),
                f"{metrics_by_configuration[configuration].frontier_vs_baseline:.2f}x",
                NOTES[configuration],
            )
            for configuration in rendered_configurations
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Compression",
                "Cosine",
                "MSE" if args.source == "artifact" else "MSE proxy",
                "Frontier vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )
    print()
    print(
        "Takeaway: frontier values above 1.00x are better than q8_0 on the modeled "
        "compression-quality trade-off. Procedural Weights is the strongest single-pillar "
        "frontier gain, and the full Blackhole stack is best overall because it adds both "
        "compression and quality at the same time."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
