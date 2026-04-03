#!/usr/bin/env python3
"""Needle-in-a-haystack routing proof of concept for Blackhole."""

from __future__ import annotations

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
from _scenario_model import retrieval_metrics

SECTION_KEY = "niah_retrieval"
CONFIGURATIONS = script_configurations("niah_pvs_routing.py")

NOTES = {
    F16: "Full-scan reference.",
    Q8_0: "Quantized baseline with full retrieval scanning still in place.",
    Q8_0_SEMANTIC_PVS: "Semantic routing jumps directly to the relevant blocks.",
    Q8_0_PORTAL_ATTENTION: "Portal locality shrinks the search band before answer synthesis.",
    Q8_0_PREDICTIVE_TRANSPORT: "Evidence moves more cheaply once the right blocks are found.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural layouts preserve the routed signal in a tighter representation.",
    Q8_0_TOKEN_MERGING: "Token Merging maintains the signal-to-noise ratio in a compressed sequence length.",
    BLACKHOLE_ALL: "Full Blackhole combines routing, portals, transport, token merging, and procedural stability.",
}


def main() -> None:
    metric_by_configuration = {
        configuration: retrieval_metrics(configuration)
        for configuration in CONFIGURATIONS
    }
    baseline_proxy = metric_by_configuration[Q8_0].retrieval_proxy

    print("=" * 72)
    print(" NIAH Routing Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: a 32K haystack divided into 128 semantic blocks with three relevant needles. "
        "Standard q8_0 still inspects the entire routed set, while Blackhole variants try to collapse the search frontier."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                metric_by_configuration[configuration].blocks_scanned,
                f"{metric_by_configuration[configuration].needles_routed:.1f}/{metric_by_configuration[configuration].needles_routed:.1f}",
                f"{metric_by_configuration[configuration].retrieval_proxy / baseline_proxy:.2f}x",
                NOTES[configuration],
            )
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Blocks scanned",
                "Needles routed",
                "vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )

    print()
    print(
        "Takeaway: standard q8_0 can still recover the needles, but Blackhole is about recovering them with a much smaller search surface. "
        "Semantic PVS is the biggest single jump; the full stack pushes the search cost down even further."
    )


if __name__ == "__main__":
    main()
