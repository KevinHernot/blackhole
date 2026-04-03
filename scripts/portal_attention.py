#!/usr/bin/env python3
"""Portal Attention proof of concept for Blackhole.

This script is intentionally a *proof of concept*: it shows how Portal Attention and the
other Blackhole variants improve on the standard ``q8_0`` prefill story using the same
shared q8_0-primary ladder.
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
    markdown_table,
    ordered_section_rows,
    render_section_overview,
    script_configurations,
)
from _scenario_model import prefill_metrics

SECTION_KEY = "prefill_context_scaling"
CONFIGURATIONS = script_configurations("portal_attention.py")

NOTES = {
    F16: "Dense fp16 reference — no routing or portal shrinkage.",
    Q8_0: "Quantized baseline with no routing or portal shrinkage yet.",
    Q8_0_SEMANTIC_PVS: "Semantic routing prunes cold regions before the portal window is built.",
    Q8_0_PORTAL_ATTENTION: "Portal frustum keeps only sinks, bridge tokens, and the active domain.",
    Q8_0_PREDICTIVE_TRANSPORT: "Transport gets lighter, but the active prefill window is still q8_0-sized.",
    Q8_0_PROCEDURAL_WEIGHTS: "Better layout helps the same q8_0 window decode a bit more cheaply.",
    Q8_0_TOKEN_MERGING: "Token Merging reduces the absolute number of tokens to scan before portals even apply.",
    BLACKHOLE_ALL: "All five pillars stack: routing plus portals plus transport plus procedural layouts plus token merging.",
}


def main() -> None:
    metric_by_configuration = {
        configuration: prefill_metrics(configuration)
        for configuration in CONFIGURATIONS
    }
    baseline_speed = metric_by_configuration[Q8_0].prefill_speed_proxy

    print("=" * 72)
    print(" Portal Attention Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: a 3-domain prompt where standard q8_0 keeps the full routed window live, "
        "while Portal Attention shrinks the active KV working set to sinks + bridge + active domain."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                metric_by_configuration[configuration].active_kv_tokens,
                f"{metric_by_configuration[configuration].kv_reduction_fraction * 100:.1f}%",
                f"{metric_by_configuration[configuration].prefill_speed_proxy:.2f}x",
                f"{metric_by_configuration[configuration].prefill_speed_proxy / baseline_speed:.2f}x",
                NOTES[configuration],
            )
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            (
                "Configuration",
                "Active KV tokens",
                "KV reduction",
                "Prefill speed proxy",
                "vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )

    print()
    print(
        "Takeaway: standard q8_0 is already a strong compressed baseline, but Blackhole starts to win once it "
        "shrinks *which* tokens stay active. Portal Attention is the first sharp jump; the full "
        "Blackhole stack compounds that win instead of stopping there."
    )


if __name__ == "__main__":
    main()
