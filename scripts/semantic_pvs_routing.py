#!/usr/bin/env python3
"""Semantic PVS routing proof of concept for Blackhole."""

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
from _scenario_model import moe_decode_metrics

SECTION_KEY = "decode_speed_moe"
CONFIGURATIONS = script_configurations("semantic_pvs_routing.py")

NOTES = {
    F16: "Dense fp16 reference.",
    Q8_0: "Quantized baseline with no Blackhole routing yet.",
    Q8_0_SEMANTIC_PVS: "Macro-routing keeps only the most relevant semantic blocks.",
    Q8_0_PORTAL_ATTENTION: "Portal windows reduce domain spill but do less macro-culling than Semantic PVS.",
    Q8_0_PREDICTIVE_TRANSPORT: "The same routed work becomes cheaper to move across layers.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural reconstruction improves the work done per surviving value read.",
    Q8_0_TOKEN_MERGING: "Merged tokens reduce the total key-set dimension before routing filters begin.",
    BLACKHOLE_ALL: "Routing, portals, transport, token merging, and procedural weights compound instead of competing.",
}


def main() -> None:
    metric_by_configuration = {
        configuration: moe_decode_metrics(configuration)
        for configuration in CONFIGURATIONS
    }
    baseline_decode = metric_by_configuration[Q8_0].decode_speed_proxy

    print("=" * 72)
    print(" Semantic PVS Routing Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: a 16K MoE decode step where standard q8_0 still examines the full routed token set, "
        "while Semantic PVS culls entire semantic blocks before value reads begin."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                metric_by_configuration[configuration].active_key_tokens,
                metric_by_configuration[configuration].values_processed,
                f"{metric_by_configuration[configuration].compute_reduction_fraction * 100:.2f}%",
                f"{metric_by_configuration[configuration].decode_speed_proxy:.2f}x",
                f"{metric_by_configuration[configuration].decode_speed_proxy / baseline_decode:.2f}x",
                NOTES[configuration],
            )
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Active key tokens",
                "Values processed",
                "Compute reduction",
                "Decode proxy",
                "vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )

    print()
    print(
        "Takeaway: standard q8_0 is a strong cache baseline, but Semantic PVS changes *which blocks survive at all*. "
        "That makes it the clearest routing upgrade on top of q8_0, and the full Blackhole stack pushes the same idea even further."
    )


if __name__ == "__main__":
    main()
