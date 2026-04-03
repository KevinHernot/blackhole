#!/usr/bin/env python3
"""Sparse-V skip-rate proof of concept for Blackhole.

The previous version measured a real Hugging Face model. This redo intentionally keeps the
script lightweight and deterministic so it can act as a clean proof of concept for how each
Blackhole configuration improves on standard ``q8_0`` at long-context dense decode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from _scenario_model import (
    DEFAULT_DENSE_CONTEXTS,
    MEASUREMENT_MODE,
    MEASURED_RUNTIME,
    average_dense_skip_rate,
    dense_decode_proxy,
    dense_skip_rate,
)

SECTION_KEY = "decode_speed_dense"
CONFIGURATIONS = script_configurations("measure_skip_rate.py")

NOTES = {
    F16: "Dense fp16 reference.",
    Q8_0: "Quantized baseline, still using the same dense gating surface throughout decode.",
    Q8_0_SEMANTIC_PVS: "Routing sharpens relevance, so more cold values can be skipped safely.",
    Q8_0_PORTAL_ATTENTION: "Portal windows keep the active domain tight, increasing skip leverage.",
    Q8_0_PREDICTIVE_TRANSPORT: "Transport gains do not change sparsity directly, but they make surviving reads cheaper.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural structure makes the same dense skip decisions more stable.",
    Q8_0_TOKEN_MERGING: "Merging tokens increases the per-token salience, making skip thresholds more effective.",
    BLACKHOLE_ALL: "All five pillars combine into the strongest skip leverage on top of q8_0.",
}


def _label_context(context_length: int) -> str:
    if context_length >= 1024:
        return f"{context_length // 1024}K"
    return str(context_length)


def _parse_contexts(argv: list[str]) -> list[int]:
    if not argv:
        return list(DEFAULT_DENSE_CONTEXTS)

    contexts: list[int] = []
    for raw_value in argv:
        try:
            context_length = int(raw_value)
        except ValueError as exc:
            raise SystemExit(
                f"Error: context lengths must be integers, got {raw_value!r}. "
                "Example: python3 scripts/measure_skip_rate.py 1024 2048 4096"
            ) from exc

        if context_length <= 0:
            raise SystemExit(f"Error: context lengths must be positive integers, got {context_length}.")

        contexts.append(context_length)

    return contexts


def _resolve_output_dir(raw: str | None) -> Path:
    if raw is None:
        return Path(__file__).parent.parent / "docs" / "threshold-ablation-logs"

    output_dir = Path(raw).expanduser()
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).parent.parent / output_dir).resolve()
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "contexts",
        nargs="*",
        help="Optional context lengths (default: shared dense context sweep).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for the proof-of-concept JSON log (default: docs/threshold-ablation-logs).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Render the tables without writing the proof-of-concept JSON log.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)
    contexts = _parse_contexts(args.contexts)

    print("=" * 72)
    print(" Sparse-V Skip-Rate Proof of Concept ")
    print("=" * 72)
    print(
        "Scenario: dense decode at increasing context lengths, with standard q8_0 as the gating baseline "
        "and Blackhole variants showing how much more inactive value mass can be skipped safely."
    )
    print()
    print(render_section_overview(SECTION_KEY, CONFIGURATIONS))
    print()

    average_by_configuration = {
        configuration: average_dense_skip_rate(configuration, contexts)
        for configuration in CONFIGURATIONS
    }
    table_rows = []
    json_rows = []
    for configuration in CONFIGURATIONS:
        rates = [dense_skip_rate(configuration, context) for context in contexts]
        average = average_by_configuration[configuration]
        row = [configuration]
        for rate in rates:
            row.append(f"{rate * 100:.1f}%")
        row.extend(
            (
                f"{average * 100:.1f}%",
                f"{dense_decode_proxy(configuration, contexts):.2f}x",
                NOTES[configuration],
            )
        )
        table_rows.append(row)
        json_rows.append(
            {
                "configuration": configuration,
                "rates": {str(context): rate for context, rate in zip(contexts, rates)},
                "average": average,
                "vs_q8_0": dense_decode_proxy(configuration, contexts),
            }
        )

    print(
        markdown_table(
            (
                "Configuration",
                *[_label_context(context) for context in contexts],
                "Average",
                "vs q8_0",
                "Why it improves q8_0",
            ),
            table_rows,
        )
    )
    print()

    summary_rows = ordered_section_rows(
        SECTION_KEY,
        {
            configuration: (
                f"{average_by_configuration[configuration] * 100:.1f}%",
                f"{dense_decode_proxy(configuration, contexts):.2f}x",
                NOTES[configuration],
            )
            for configuration in CONFIGURATIONS
        },
    )
    print(
        markdown_table(
            ("Configuration", "Average skip leverage", "Decode proxy vs q8_0", "Interpretation"),
            summary_rows,
        )
    )

    if not args.no_write:
        output_dir = _resolve_output_dir(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "skip_rate_proof_of_concept.json"
        output_file.write_text(
            json.dumps(
                {
                    "proof_of_concept": True,
                    "measurement_mode": MEASUREMENT_MODE,
                    "measured_runtime": MEASURED_RUNTIME,
                    "contexts": contexts,
                    "results": json_rows,
                },
                indent=2,
            )
            + "\n"
        )
        print()
        print(f"Saved proof-of-concept data to: {output_file}")


if __name__ == "__main__":
    main()
