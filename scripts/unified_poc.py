#!/usr/bin/env python3
"""Unified Blackhole proof of concept.

This script stacks the same q8_0-primary ladder used everywhere else and prints
a compact summary of how each Blackhole idea improves on the standard
``q8_0`` architecture.
"""

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
    RESULT_SECTION_ORDER,
    markdown_table,
    ordered_section_rows,
    render_section_overview,
    script_configurations,
    section_profile,
)
from _scenario_model import top_of_tree_summary

SECTION_KEY = "top_of_tree_results"
CONFIGURATIONS = script_configurations("unified_poc.py")

TOP_OF_TREE_NOTES = {
    F16: "Full-precision dense reference.",
    Q8_0: "Quantized baseline before Blackhole-specific ideas.",
    Q8_0_SEMANTIC_PVS: "Adds macro-routing on top of q8_0.",
    Q8_0_PORTAL_ATTENTION: "Adds domain-local portal windows on top of q8_0.",
    Q8_0_PREDICTIVE_TRANSPORT: "Adds lighter layer-to-layer transport on top of q8_0.",
    Q8_0_PROCEDURAL_WEIGHTS: "Improves the compression-quality frontier on top of q8_0.",
    Q8_0_TOKEN_MERGING: "Adds sequence-dimension 'Greedy Meshing' on top of q8_0.",
    BLACKHOLE_ALL: "Stacks all five Blackhole pillars on top of q8_0.",
}


def main() -> None:
    print("=" * 72)
    print(" Unified Blackhole Proof of Concept ")
    print("=" * 72)
    print(
        "This is the stitched-together Blackhole story: every row uses the same q8_0-primary ladder, "
        "and every delta is explained relative to the same standard q8_0 baseline."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (*top_of_tree_summary(configuration), TOP_OF_TREE_NOTES[configuration])
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Compression",
                "Prefill proxy",
                "MoE decode proxy",
                "NIAH proxy",
                "Transport proxy",
                "Quality proxy",
                "Why it improves q8_0",
            ),
            rows,
        )
    )
    print()

    catalog_rows = [
        (section_profile(section_key).title, ", ".join(section_profile(section_key).configurations))
        for section_key in RESULT_SECTION_ORDER
    ]
    print("Section coverage catalog")
    print(markdown_table(("Result section", "Configuration ladder"), catalog_rows))
    print()
    print(
        "Takeaway: q8_0 is the common compressed baseline. Blackhole is not a different baseline; it is the set of ideas "
        "that make q8_0 better, one pillar at a time and then all five together."
    )


if __name__ == "__main__":
    main()
