#!/usr/bin/env python3
"""Blackhole Needle-In-A-Haystack proof of concept.

This script intentionally avoids pretending that a native Blackhole runtime already exists.
Instead, it runs a deterministic retrieval proof of concept over the same
q8_0-primary user-facing configurations used by every other Python script in
this repository, always with standard ``q8_0`` as the common baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    section_configurations,
    validate_configurations,
)
from _scenario_model import retrieval_probability
from _scenario_model import MEASUREMENT_MODE, MEASURED_RUNTIME

SEED = 42
DEFAULT_NIAH_CONFIGURATIONS = script_configurations("niah_test.py")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "niah_results_poc"

CONFIGURATION_NOTES = {
    F16: "Full-precision dense reference.",
    Q8_0: "Quantized baseline before Blackhole-specific improvements.",
    Q8_0_SEMANTIC_PVS: "Semantic routing narrows the retrieval frontier before answer synthesis.",
    Q8_0_PORTAL_ATTENTION: "Portal locality keeps the active domain tight around the answer region.",
    Q8_0_PREDICTIVE_TRANSPORT: "Transport stabilization helps the right evidence survive the decode path.",
    Q8_0_PROCEDURAL_WEIGHTS: "Procedural layouts preserve compressed recall more faithfully than plain q8_0.",
    Q8_0_TOKEN_MERGING: "Sequence-dimension merging increases per-token salience before synthesis.",
    BLACKHOLE_ALL: "All five Blackhole pillars compound on top of q8_0.",
}

SYNTHETIC_NUMBER_MIN = 1_000_000
SYNTHETIC_NUMBER_MAX = 9_999_999
WRONG_NUMBER_RANDOM_ATTEMPTS = 256


@dataclass(frozen=True)
class NeedleSpec:
    value: str
    depth_pct: float


@dataclass
class TrialResult:
    expected: str
    response: str
    found: bool
    expected_probability: float = 0.0
    needle_depth_pct: float = 0.0
    context_length: int = 0


@dataclass
class ConfigResult:
    mode: str
    context_length: int
    cache_type: str
    needle_depth_pct: float = 0.5
    needle_count: int = 1
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def configuration(self) -> str:
        return self.cache_type

    @property
    def accuracy_pct(self) -> float:
        if not self.trials:
            return 0.0
        return sum(1 for trial in self.trials if trial.found) / len(self.trials) * 100.0

    @property
    def expected_accuracy_pct(self) -> float:
        if not self.trials:
            return 0.0
        return sum(trial.expected_probability for trial in self.trials) / len(self.trials) * 100.0

    @property
    def passed(self) -> bool:
        return all(trial.found for trial in self.trials)


def _split_csv(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_requested_configurations(raw: str) -> list[str]:
    requested = _split_csv(raw) or list(DEFAULT_NIAH_CONFIGURATIONS)
    return list(validate_configurations(requested))


def _ordered_result_configurations(results: list[ConfigResult]) -> list[str]:
    present = list(validate_configurations(result.cache_type for result in results))
    preferred = [
        configuration
        for configuration in section_configurations("niah_retrieval")
        if configuration in present
    ]
    extras = [configuration for configuration in present if configuration not in preferred]
    return preferred + extras


def _seed_for(*parts: object) -> int:
    payload = "|".join(str(part) for part in (SEED, *parts)).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:16], 16)


def _stable_roll(*parts: object) -> float:
    return random.Random(_seed_for("roll", *parts)).random()


def _synthetic_number_space() -> int:
    return SYNTHETIC_NUMBER_MAX - SYNTHETIC_NUMBER_MIN + 1


def _draw_synthetic_number(rng: random.Random) -> str:
    return str(rng.randint(SYNTHETIC_NUMBER_MIN, SYNTHETIC_NUMBER_MAX))


def _make_magic_number(*parts: object) -> str:
    rng = random.Random(_seed_for("value", *parts))
    return _draw_synthetic_number(rng)


def _wrong_number(forbidden: set[str], *parts: object) -> str:
    if len(forbidden) >= _synthetic_number_space():
        raise ValueError("Forbidden values exhaust the available synthetic number space.")

    rng = random.Random(_seed_for("wrong", *parts))
    for _ in range(min(WRONG_NUMBER_RANDOM_ATTEMPTS, _synthetic_number_space())):
        candidate = _draw_synthetic_number(rng)
        if candidate not in forbidden:
            return candidate

    start = rng.randint(SYNTHETIC_NUMBER_MIN, SYNTHETIC_NUMBER_MAX)
    for offset in range(_synthetic_number_space()):
        candidate = str(
            SYNTHETIC_NUMBER_MIN + ((start - SYNTHETIC_NUMBER_MIN + offset) % _synthetic_number_space())
        )
        if candidate not in forbidden:
            return candidate

    raise ValueError("Unable to generate a fallback synthetic number outside the forbidden set.")


def _simulate_single_trial(configuration: str, context_length: int, depth_pct: int) -> TrialResult:
    expected = _make_magic_number(configuration, context_length, depth_pct, "single")
    probability = retrieval_probability(configuration, context_length, float(depth_pct), "single")
    found = _stable_roll(configuration, context_length, depth_pct, "single") < probability
    response = expected if found else _wrong_number({expected}, configuration, context_length, depth_pct, "single")
    return TrialResult(
        expected=expected,
        response=response,
        found=found,
        expected_probability=probability,
        needle_depth_pct=depth_pct / 100.0,
        context_length=context_length,
    )


def _simulate_multi_key_trial(
    configuration: str,
    context_length: int,
    num_distractors: int,
) -> TrialResult:
    expected = _make_magic_number(configuration, context_length, num_distractors, "multi-key", "real")
    probability = retrieval_probability(
        configuration,
        context_length,
        50.0,
        "multi-key",
        num_distractors=num_distractors,
    )
    found = _stable_roll(configuration, context_length, num_distractors, "multi-key") < probability
    response = expected if found else _wrong_number(
        {expected}, configuration, context_length, num_distractors, "multi-key"
    )
    return TrialResult(
        expected=expected,
        response=response,
        found=found,
        expected_probability=probability,
        needle_depth_pct=0.5,
        context_length=context_length,
    )


def _simulate_multi_value_result(
    configuration: str,
    context_length: int,
    value_count: int,
) -> ConfigResult:
    needles: list[NeedleSpec] = []
    hits: list[bool] = []
    probabilities: list[float] = []

    for index in range(value_count):
        depth_pct = ((index + 1) / (value_count + 1)) * 100.0
        value = _make_magic_number(configuration, context_length, value_count, index, "multi-value")
        probability = retrieval_probability(
            configuration,
            context_length,
            depth_pct,
            "multi-value",
            value_count=value_count,
        )
        found = _stable_roll(configuration, context_length, value_count, index, "multi-value") < probability
        needles.append(NeedleSpec(value=value, depth_pct=depth_pct))
        hits.append(found)
        probabilities.append(probability)

    found_values = [needle.value for needle, hit in zip(needles, hits) if hit]
    if found_values:
        response = ", ".join(found_values)
    else:
        response = _wrong_number({needle.value for needle in needles}, configuration, context_length, value_count, "multi-value")

    result = ConfigResult(
        mode="multi-value",
        context_length=context_length,
        cache_type=configuration,
        needle_count=value_count,
    )
    for needle, hit, probability in zip(needles, hits, probabilities):
        result.trials.append(
            TrialResult(
                expected=needle.value,
                response=response,
                found=hit,
                expected_probability=probability,
                needle_depth_pct=needle.depth_pct / 100.0,
                context_length=context_length,
            )
        )
    return result


def run_single_mode(args: argparse.Namespace) -> list[ConfigResult]:
    context_lengths = [int(value) for value in _split_csv(args.depths)]
    depth_sweep = [int(value) for value in _split_csv(args.depths_sweep)]

    results: list[ConfigResult] = []
    print(f"Single-needle proof of concept across {len(args.resolved_configurations)} configurations")
    for configuration in args.resolved_configurations:
        for context_length in context_lengths:
            hits = 0
            for depth_pct in depth_sweep:
                trial = _simulate_single_trial(configuration, context_length, depth_pct)
                result = ConfigResult(
                    mode="single",
                    context_length=context_length,
                    cache_type=configuration,
                    needle_depth_pct=depth_pct / 100.0,
                )
                result.trials.append(trial)
                results.append(result)
                hits += int(trial.found)
            ctx_label = f"{context_length // 1024}K" if context_length >= 1024 else str(context_length)
            print(f"  {configuration:<36} ctx={ctx_label:<4} -> {hits}/{len(depth_sweep)} hits")
    return results


def run_multi_key_mode(args: argparse.Namespace) -> list[ConfigResult]:
    context_lengths = [int(value) for value in _split_csv(args.depths)]
    num_distractors = args.num_distractors

    results: list[ConfigResult] = []
    print(
        f"Multi-key proof of concept across {len(args.resolved_configurations)} configurations "
        f"with {num_distractors} distractors"
    )
    for configuration in args.resolved_configurations:
        for context_length in context_lengths:
            trial = _simulate_multi_key_trial(configuration, context_length, num_distractors)
            result = ConfigResult(
                mode="multi-key",
                context_length=context_length,
                cache_type=configuration,
                needle_depth_pct=0.5,
                needle_count=1 + num_distractors,
            )
            result.trials.append(trial)
            results.append(result)
            ctx_label = f"{context_length // 1024}K" if context_length >= 1024 else str(context_length)
            status = "hit" if trial.found else "miss"
            print(f"  {configuration:<36} ctx={ctx_label:<4} -> {status}")
    return results


def run_multi_value_mode(args: argparse.Namespace) -> list[ConfigResult]:
    context_lengths = [int(value) for value in _split_csv(args.depths)]
    value_counts = [int(value) for value in _split_csv(args.value_counts)]

    results: list[ConfigResult] = []
    print(f"Multi-value proof of concept across {len(args.resolved_configurations)} configurations")
    for configuration in args.resolved_configurations:
        for context_length in context_lengths:
            for value_count in value_counts:
                result = _simulate_multi_value_result(configuration, context_length, value_count)
                results.append(result)
                ctx_label = f"{context_length // 1024}K" if context_length >= 1024 else str(context_length)
                hits = sum(1 for trial in result.trials if trial.found)
                print(
                    f"  {configuration:<36} ctx={ctx_label:<4} values={value_count:<2} -> {hits}/{value_count} hits"
                )
    return results


def _build_heatmap_table(results: list[ConfigResult], configuration: str) -> str:
    configuration_results = [result for result in results if result.cache_type == configuration]
    if not configuration_results:
        return f"## Single Needle Retrieval: {configuration}\n\n(no results)\n"

    depths = sorted({int(result.needle_depth_pct * 100) for result in configuration_results})
    lengths = sorted({result.context_length for result in configuration_results})
    lookup = {
        (int(result.needle_depth_pct * 100), result.context_length): result.passed
        for result in configuration_results
    }

    header = "| Depth |" + "".join(
        f" {f'{length // 1024}K' if length >= 1024 else length:<5}|" for length in lengths
    )
    separator = "|---|" + "".join("---|" for _ in lengths)
    lines = [f"## Single Needle Retrieval: {configuration}", "", header, separator]

    for depth in depths:
        row = f"| {depth:>3}% |"
        for length in lengths:
            cell = " ✅ " if lookup.get((depth, length)) else " ❌ "
            row += f"{cell}|"
        lines.append(row)

    sample_hits = sum(1 for result in configuration_results if result.passed)
    expected_average = sum(result.expected_accuracy_pct for result in configuration_results) / len(
        configuration_results
    )
    lines.extend(
        [
            "",
            (
                f"**Illustrative sample: {sample_hits}/{len(configuration_results)} hits "
                f"({sample_hits / len(configuration_results) * 100.0:.1f}%) "
                f"| Expected mean hit rate: {expected_average:.1f}%**"
            ),
        ]
    )
    return "\n".join(lines)


def _build_multi_key_table(results: list[ConfigResult]) -> str:
    configurations = _ordered_result_configurations(results)
    lengths = sorted({result.context_length for result in results})

    header = "| Configuration |" + "".join(
        f" {f'{length // 1024}K' if length >= 1024 else length:<5}|" for length in lengths
    )
    separator = "|---|" + "".join("---|" for _ in lengths)
    lines = ["## Multi-Key Retrieval (MK-NIAH)", "", header, separator]

    lookup = {
        (result.cache_type, result.context_length): result.expected_accuracy_pct for result in results
    }
    for configuration in configurations:
        row = f"| {configuration} |"
        for length in lengths:
            row += f" {lookup.get((configuration, length), 0.0):>5.1f}% |"
        lines.append(row)
    return "\n".join(lines)


def _build_multi_value_table(results: list[ConfigResult]) -> str:
    configurations = _ordered_result_configurations(results)
    lengths = sorted({result.context_length for result in results})
    value_counts = sorted({result.needle_count for result in results})

    parts = ["## Multi-Value Retrieval (MV-NIAH)", ""]
    for configuration in configurations:
        header = "| Values |" + "".join(
            f" {f'{length // 1024}K' if length >= 1024 else length:<7}|" for length in lengths
        )
        separator = "|---|" + "".join("---|" for _ in lengths)
        parts.extend([f"### {configuration}", "", header, separator])

        lookup = {
            (result.needle_count, result.context_length): result.expected_accuracy_pct
            for result in results
            if result.cache_type == configuration
        }
        for value_count in value_counts:
            row = f"| {value_count:>6} |"
            for length in lengths:
                pct = lookup.get((value_count, length), 0.0)
                row += f" {pct:>5.1f}% |"
            parts.append(row)
        parts.append("")
    return "\n".join(parts)


def _build_accuracy_summary(results: list[ConfigResult]) -> str:
    configurations = _ordered_result_configurations(results)
    averages = {
        configuration: sum(
            result.expected_accuracy_pct for result in results if result.cache_type == configuration
        )
        / max(1, sum(1 for result in results if result.cache_type == configuration))
        for configuration in configurations
    }
    baseline_average = averages.get(Q8_0, 0.0)

    rows = ordered_section_rows(
        "niah_retrieval",
        {
            configuration: (
                f"{averages[configuration]:.1f}%",
                "baseline" if configuration == Q8_0 else f"{averages[configuration] - baseline_average:+.1f} pts",
                CONFIGURATION_NOTES[configuration],
            )
            for configuration in configurations
        },
    )
    return "\n".join(
        [
            "## Expected retrieval summary",
            "",
            markdown_table(
                ("Configuration", "Expected average accuracy", "vs q8_0", "Why it improves q8_0"),
                rows,
            ),
        ]
    )


def build_output(results: list[ConfigResult], model_name: str, mode: str) -> str:
    configurations = _ordered_result_configurations(results)
    parts = [
        f"# Blackhole NIAH Proof of Concept: {model_name}",
        f"Mode: {mode} | Seed: {SEED} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        render_section_overview("niah_retrieval", configurations),
        "",
        "Any hit/miss examples below are illustrative deterministic draws from the scenario model.",
        "All ranking tables and average-accuracy summaries use expected hit rates from `retrieval_probability()` so the retrieval story is not driven by single-sample noise.",
        "",
    ]

    if mode == "single":
        for configuration in configurations:
            parts.append(_build_heatmap_table(results, configuration))
            parts.append("")
    elif mode == "multi-key":
        parts.append(_build_multi_key_table(results))
        parts.append("")
    elif mode == "multi-value":
        parts.append(_build_multi_value_table(results))
        parts.append("")

    parts.append(_build_accuracy_summary(results))
    return "\n".join(parts)


def save_results(
    results: list[ConfigResult],
    model_name: str,
    mode: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"niah_{mode}_{timestamp}.json"
    json_payload = {
        "proof_of_concept": True,
        "measurement_mode": MEASUREMENT_MODE,
        "measured_runtime": MEASURED_RUNTIME,
        "model": model_name,
        "mode": mode,
        "seed": SEED,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "section_profile": list(section_configurations("niah_retrieval")),
        "results": [
            {
                "mode": result.mode,
                "context_length": result.context_length,
                "configuration": result.configuration,
                "cache_type": result.cache_type,
                "needle_depth_pct": result.needle_depth_pct,
                "needle_count": result.needle_count,
                "accuracy_pct": result.accuracy_pct,
                "expected_accuracy_pct": result.expected_accuracy_pct,
                "passed": result.passed,
                "trials": [
                    {
                        "expected": trial.expected,
                        "response": trial.response,
                        "found": trial.found,
                        "expected_probability": trial.expected_probability,
                        "needle_depth_pct": trial.needle_depth_pct,
                        "context_length": trial.context_length,
                    }
                    for trial in result.trials
                ],
            }
            for result in results
        ],
    }
    json_path.write_text(json.dumps(json_payload, indent=2) + "\n")

    markdown_path = output_dir / f"niah_{mode}_{timestamp}.md"
    markdown_path.write_text(build_output(results, model_name, mode) + "\n")
    return json_path, markdown_path


def _resolve_output_dir(raw: str | None) -> Path:
    if raw is None:
        return DEFAULT_OUTPUT_DIR

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blackhole NIAH proof-of-concept runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Modes:
              single       Sweep needle depth x context length
              multi-key    Real needle + distractors at fixed depth
              multi-value  Multiple values for the same key across the full context

            Examples:
              python3 scripts/niah_test.py
              python3 scripts/niah_test.py --mode single
              python3 scripts/niah_test.py --mode multi-key --num-distractors 5
              python3 scripts/niah_test.py --mode multi-value --value-counts 2,4,8
            """
        ),
    )
    parser.add_argument("llama_dir", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument("model_path", nargs="?", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode",
        choices=["single", "multi-key", "multi-value"],
        default="single",
        help="Retrieval proof-of-concept mode (default: %(default)s)",
    )
    parser.add_argument(
        "--depths",
        default="4096,8192,16384,32768",
        help="Comma-separated context lengths in tokens (default: %(default)s)",
    )
    parser.add_argument(
        "--depths-sweep",
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Comma-separated depth percentages for single mode (default: %(default)s)",
    )
    parser.add_argument(
        "--configs",
        default=",".join(DEFAULT_NIAH_CONFIGURATIONS),
        help=(
            "Comma-separated Blackhole configuration labels to test "
            f"(default: {','.join(DEFAULT_NIAH_CONFIGURATIONS)})"
        ),
    )
    parser.add_argument("--cache-types", dest="configs", help=argparse.SUPPRESS)
    parser.add_argument(
        "--num-distractors",
        type=int,
        default=3,
        help="Number of distractors for multi-key mode (default: %(default)s)",
    )
    parser.add_argument(
        "--value-counts",
        default="2,4,8",
        help="Comma-separated value counts for multi-value mode (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory for markdown and JSON output (default: repo-root niah_results_poc)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print additional notes, including deprecated runtime arguments if supplied.",
    )

    parsed = parser.parse_args(argv)
    parsed.cache_types = parsed.configs
    return parsed


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.resolved_configurations = _parse_requested_configurations(args.configs)

    if args.verbose and (args.llama_dir or args.model_path):
        print(
            "Note: deprecated llama runtime arguments were provided, but this script now runs as a pure proof of concept."
        )

    model_name = Path(args.model_path).stem if args.model_path else "blackhole-proof-of-concept"
    print("=" * 60)
    print("  Blackhole NIAH Proof of Concept")
    print(f"  Mode: {args.mode}")
    print(f"  Model label: {model_name}")
    print(f"  Configurations: {args.resolved_configurations}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    if args.mode == "single":
        results = run_single_mode(args)
    elif args.mode == "multi-key":
        results = run_multi_key_mode(args)
    else:
        results = run_multi_value_mode(args)

    output = build_output(results, model_name, args.mode)
    print(f"\n{output}\n")

    json_path, markdown_path = save_results(
        results,
        model_name,
        args.mode,
        _resolve_output_dir(args.output_dir),
    )
    print("Saved results:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
