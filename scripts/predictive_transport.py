#!/usr/bin/env python3
"""Predictive Transport proof of concept for Blackhole."""

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
from _scenario_model import transport_metrics

SECTION_KEY = "speed_optimization_journey"
CONFIGURATIONS = script_configurations("predictive_transport.py")

STABILITY_NOTES = {
    F16: "Native reference — no transport approximation.",
    Q8_0: "Quantized baseline, still sending dense layer payloads.",
    Q8_0_SEMANTIC_PVS: "Fewer routed blocks means less payload before transport even starts.",
    Q8_0_PORTAL_ATTENTION: "Smaller active windows cut the amount of state that must cross layers.",
    Q8_0_PREDICTIVE_TRANSPORT: "Predicted deltas replace full payload copies while keeping cosine ≈ 0.999.",
    Q8_0_PROCEDURAL_WEIGHTS: "Structured residuals compress more cleanly than plain q8_0 deltas.",
    Q8_0_TOKEN_MERGING: "Merging tokens shrinks the absolute transport payload volume before prediction deltas are even computed.",
    BLACKHOLE_ALL: "All five pillars align: less state, fewer tokens, better deltas, better reconstruction.",
}


def main() -> None:
    metric_by_configuration = {
        configuration: transport_metrics(configuration)
        for configuration in CONFIGURATIONS
    }

    print("=" * 72)
    print(" Predictive Transport Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: a 128-layer pipeline where standard q8_0 still copies full per-layer payloads, "
        "while Predictive Transport ships deltas and lets downstream layers reconstruct the next state."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                f"{metric_by_configuration[configuration].transported_volume_gb:.2f} GB",
                f"{metric_by_configuration[configuration].reduction_vs_f16 * 100:.1f}%",
                f"{metric_by_configuration[configuration].speed_proxy_vs_baseline:.2f}x",
                STABILITY_NOTES[configuration],
            )
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Transported volume",
                "Reduction vs f16",
                "Speed proxy vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )

    print()
    print(
        "Takeaway: standard q8_0 already moves less state than fp16, but Predictive Transport is where Blackhole "
        "starts shrinking the *handoff itself*. The full stack wins because routing and portals reduce what must move, "
        "and predictive deltas reduce how much each surviving step costs."
    )


if __name__ == "__main__":
    main()
