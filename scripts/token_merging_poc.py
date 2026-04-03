#!/usr/bin/env python3
"""Token Merging (Greedy Meshing) proof of concept for Blackhole."""

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
from _scenario_model import mechanics, prefill_metrics

SECTION_KEY = "sequence_compression_tome"
CONFIGURATIONS = script_configurations("token_merging_poc.py")

NOTES = {
    F16: "Uncompressed dense reference.",
    Q8_0: "8-bit baseline before sequence-dimension ideas.",
    Q8_0_SEMANTIC_PVS: "Prunes blocks, but does not fuse tokens within blocks.",
    Q8_0_PORTAL_ATTENTION: "Shrinks the active window, but keeps it dense.",
    Q8_0_PREDICTIVE_TRANSPORT: "Lighter layer hand-offs, sequence remains standard q8_0.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural reconstruction helps quality, not sequence length.",
    Q8_0_TOKEN_MERGING: "Fuses redundant tokens, shrinking the sequence by ~35%.",
    BLACKHOLE_ALL: "All 5 pillars stack: bits compressed, sequence merged, irrelevant blocks routed.",
}

# The user requested an example scenario of 16384 tokens.
# We'll use a 16K "original" anchor for this POC.
ORIGINAL_LENGTH = 16_384


def main() -> None:
    print("=" * 72)
    print(" Token Merging (Greedy Meshing) Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: Adapting 3D rendering 'Greedy Meshing' to LLMs. "
        "Instead of just shrinking the bit-depth of the cache, "
        "we geometrically merge redundant adjacent tokens to shrink the sequence length."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    baseline_metrics = prefill_metrics(Q8_0)
    baseline_speed = baseline_metrics.prefill_speed_proxy

    values_by_config = {}
    for config in CONFIGURATIONS:
        cfg_mechanics = mechanics(config)
        metrics = prefill_metrics(config)
        
        # Token merging reduction factor is 0.35 in blackhole_core.scenario_model
        # When cfg_mechanics.token_merging == 1.0, ratio is multiplied by 0.65
        reduction_multiplier = 1.0 - (0.35 * cfg_mechanics.token_merging)
        merged_length = int(ORIGINAL_LENGTH * reduction_multiplier)
        reduction_pct = f"{(1.0 - reduction_multiplier) * 100:.0f}%"
        
        values_by_config[config] = (
            ORIGINAL_LENGTH,
            merged_length,
            reduction_pct,
            f"{metrics.prefill_speed_proxy:.2f}x",
            f"{metrics.prefill_speed_proxy / baseline_speed:.2f}x",
            NOTES[config],
        )

    rows = ordered_section_rows(SECTION_KEY, values_by_config)
    print(
        markdown_table(
            (
                "Configuration",
                "Original length",
                "Merged length",
                "Sequence reduction",
                "Prefill proxy",
                "vs q8_0",
                "Why it improves q8_0",
            ),
            rows,
        )
    )

    print()
    print(
        "Takeaway: Token Merging is the sequence-dimension equivalent of 3D 'Greedy Meshing'. "
        "By merging similar adjacent tokens, we reduce the payload for attention and transport, "
        "stacking multiplicatively with the rest of the Blackhole working-state reductions."
    )


if __name__ == "__main__":
    main()
