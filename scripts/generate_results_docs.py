#!/usr/bin/env python3
"""Generate result docs from Blackhole proof-of-concept outputs."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import importlib.util
import inspect
import io
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DOCS_DIR = PROJECT_ROOT / "docs"
NIAH_DIR = PROJECT_ROOT / "niah_results_poc"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blackhole_core.comparison_profiles import (  # noqa: E402
    BLACKHOLE_ALL,
    Q8_0,
    markdown_table,
)
from blackhole_core.run_manifest import (  # noqa: E402
    RunManifest,
    discover_latest_niah_artifacts,
    utc_now_iso,
)
from blackhole_core.scenario_model import (  # noqa: E402
    compression_quality_metrics,
    kl_divergence_metrics,
    long_context_perplexity_metrics,
    moe_decode_metrics,
    prefill_metrics,
    transport_metrics,
)

SECTION_SCRIPTS = (
    ("Unified Top-of-Tree Summary", "unified_poc", ()),
    ("Portal Attention — Prefill Context Scaling", "portal_attention", ()),
    ("Semantic PVS Routing — MoE Decode Speed", "semantic_pvs_routing", ()),
    ("Predictive Transport — Layer Handoff Efficiency", "predictive_transport", ()),
    ("Sparse-V Skip-Rate — Dense Decode", "measure_skip_rate", ("--no-write",)),
    ("Token Merging (Greedy Meshing) — Sequence Compression", "token_merging_poc", ()),
)

QUALITY_SECTION_SCRIPTS = (
    ("KL Divergence vs f16", "kl_divergence_vs_f16"),
    ("Long-Context Perplexity (Primary Quality Metric)", "long_context_perplexity"),
    ("Compression Quality (Python Prototype)", "compression_quality"),
)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_script_module(module_name: str):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"blackhole_{module_name}_docs", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"blackhole_{module_name}_docs"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _capture_script_output(module_name: str, argv: tuple[str, ...] = ()) -> str:
    module = _load_script_module(module_name)
    main = getattr(module, "main")
    signature = inspect.signature(main)
    stream = io.StringIO()
    with redirect_stdout(stream):
        if signature.parameters:
            main(list(argv))
        else:
            main()
    return stream.getvalue().strip()


def _strip_banner(output: str) -> str:
    lines = output.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) >= 3 and set(lines[0].strip()) == {"="} and set(lines[2].strip()) == {"="}:
        lines = lines[3:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _strip_leading_heading(output: str) -> str:
    lines = output.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines).strip()


def _latest_niah_averages(json_path: Path) -> dict[str, float]:
    payload = json.loads(json_path.read_text())
    aggregates: dict[str, tuple[float, int]] = {}
    for result in payload.get("results", []):
        configuration = str(result.get("configuration") or result.get("cache_type"))
        total, count = aggregates.get(configuration, (0.0, 0))
        aggregates[configuration] = (total + float(result["expected_accuracy_pct"]), count + 1)
    return {
        configuration: total / count
        for configuration, (total, count) in aggregates.items()
        if count
    }


def _quality_summary(quality_source: str, quality_artifact: Path | None) -> tuple[str, dict[str, float | None]]:
    if quality_source == "artifact":
        if quality_artifact is None:
            raise SystemExit("Error: --quality-artifact is required when --quality-source artifact.")
        from blackhole_core.measured_quality import load_measured_quality_artifact

        artifact = load_measured_quality_artifact(quality_artifact)
        metrics_by_configuration = dict(artifact.measurements)
        return (
            (
                f"{artifact.metadata.evidence_tier.value} artifact from {artifact.metadata.corpus_name} "
                f"({artifact.metadata.created_at_utc})"
            ),
            {
                "q8_0_kld": metrics_by_configuration[Q8_0].mean_kld if Q8_0 in metrics_by_configuration else None,
                "blackhole_kld": (
                    metrics_by_configuration[BLACKHOLE_ALL].mean_kld
                    if BLACKHOLE_ALL in metrics_by_configuration
                    else None
                ),
                "q8_0_ppl": (
                    metrics_by_configuration[Q8_0].long_context_perplexity
                    if Q8_0 in metrics_by_configuration
                    else None
                ),
                "blackhole_ppl": (
                    metrics_by_configuration[BLACKHOLE_ALL].long_context_perplexity
                    if BLACKHOLE_ALL in metrics_by_configuration
                    else None
                ),
                "q8_0_frontier": (
                    metrics_by_configuration[Q8_0].frontier_vs_baseline
                    if Q8_0 in metrics_by_configuration
                    else None
                ),
                "blackhole_frontier": (
                    metrics_by_configuration[BLACKHOLE_ALL].frontier_vs_baseline
                    if BLACKHOLE_ALL in metrics_by_configuration
                    else None
                ),
            },
        )

    return (
        "proxy scenario-model outputs",
        {
            "q8_0_kld": kl_divergence_metrics(Q8_0).mean_kld,
            "blackhole_kld": kl_divergence_metrics(BLACKHOLE_ALL).mean_kld,
            "q8_0_ppl": long_context_perplexity_metrics(Q8_0).ppl_proxy_32k,
            "blackhole_ppl": long_context_perplexity_metrics(BLACKHOLE_ALL).ppl_proxy_32k,
            "q8_0_frontier": compression_quality_metrics(Q8_0).frontier_vs_baseline,
            "blackhole_frontier": compression_quality_metrics(BLACKHOLE_ALL).frontier_vs_baseline,
        },
    )


def _format_metric(value: float | None, *, suffix: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}{suffix}"


def _build_results_doc(manifest: RunManifest, quality_source_label: str) -> str:
    parts = [
        "# Blackhole Proof-of-Concept Results",
        "",
        f"> **Generated at (UTC):** {manifest.generated_at_utc}",
        f"> **Baseline:** standard `{manifest.baseline}`",
        f"> **Seed:** {manifest.seed}",
        f"> **Quality source:** {quality_source_label}",
    ]
    if manifest.commit_sha:
        parts.append(f"> **Git commit:** `{manifest.commit_sha}`")
    if manifest.pytest_status:
        parts.append(f"> **Verification:** `pytest -q tests` -> **{manifest.pytest_status}**")
    parts.extend(
        [
            "",
            "## Provenance",
            "",
            "Evidence tiers in this repo are intentionally split into `proxy`, `measured_offline`, "
            "`measured_model`, and `runtime_benchmark`. These generated docs report the highest tier "
            "actually provided for each section instead of mixing copied numbers from different runs.",
            "",
            "## Latest NIAH Artifacts",
            "",
        ]
    )
    if manifest.niah_artifacts:
        for artifact in manifest.niah_artifacts:
            parts.append(
                f"- `{artifact.mode}`: `{_display_path(artifact.json_path)}` and "
                f"`{_display_path(artifact.markdown_path)}`"
            )
    else:
        parts.append("- No NIAH artifacts were found.")

    section_number = 1
    for title, module_name, argv in SECTION_SCRIPTS:
        parts.extend(
            [
                "",
                f"## {section_number}. {title}",
                "",
                _strip_banner(_capture_script_output(module_name, argv)),
            ]
        )
        section_number += 1

    for artifact in manifest.niah_artifacts:
        parts.extend(
            [
                "",
                f"## {section_number}. NIAH Retrieval — {artifact.mode}",
                "",
                f"Source artifact: `{_display_path(artifact.markdown_path)}`",
                "",
                _strip_leading_heading(artifact.markdown_path.read_text().strip()),
            ]
        )
        section_number += 1

    quality_args: tuple[str, ...] = ()
    if manifest.quality_source == "artifact" and manifest.quality_artifact is not None:
        quality_args = (
            "--source",
            "artifact",
            "--artifact",
            str(manifest.quality_artifact),
        )

    for title, module_name in QUALITY_SECTION_SCRIPTS:
        parts.extend(
            [
                "",
                f"## {section_number}. {title}",
                "",
                _strip_banner(_capture_script_output(module_name, quality_args)),
            ]
        )
        section_number += 1

    parts.extend(
        [
            "",
            "## Generated With",
            "",
            "```bash",
            "python3 scripts/generate_results_docs.py",
            "```",
        ]
    )
    return "\n".join(parts).rstrip() + "\n"


def _build_summary_doc(
    manifest: RunManifest,
    quality_source_label: str,
    quality_metrics: dict[str, float | None],
) -> str:
    prefill_q8_0 = prefill_metrics(Q8_0).prefill_speed_proxy
    prefill_blackhole = prefill_metrics(BLACKHOLE_ALL).prefill_speed_proxy
    moe_q8_0 = moe_decode_metrics(Q8_0).decode_speed_proxy
    moe_blackhole = moe_decode_metrics(BLACKHOLE_ALL).decode_speed_proxy
    transport_q8_0 = transport_metrics(Q8_0).speed_proxy_vs_baseline
    transport_blackhole = transport_metrics(BLACKHOLE_ALL).speed_proxy_vs_baseline

    niah_averages = {
        artifact.mode: _latest_niah_averages(artifact.json_path)
        for artifact in manifest.niah_artifacts
    }

    metric_rows = [
        (
            "Prefill speed proxy",
            f"{prefill_q8_0:.2f}x",
            f"{prefill_blackhole:.2f}x",
            f"{prefill_blackhole / prefill_q8_0:.2f}x",
        ),
        (
            "MoE decode proxy",
            f"{moe_q8_0:.2f}x",
            f"{moe_blackhole:.2f}x",
            f"{moe_blackhole / moe_q8_0:.2f}x",
        ),
        (
            "Transport speed proxy",
            f"{transport_q8_0:.2f}x",
            f"{transport_blackhole:.2f}x",
            f"{transport_blackhole / transport_q8_0:.2f}x",
        ),
    ]

    if "single" in niah_averages and Q8_0 in niah_averages["single"] and BLACKHOLE_ALL in niah_averages["single"]:
        single_q8_0 = niah_averages["single"][Q8_0]
        single_blackhole = niah_averages["single"][BLACKHOLE_ALL]
        metric_rows.append(
            (
                "NIAH single-needle expected accuracy",
                f"{single_q8_0:.1f}%",
                f"{single_blackhole:.1f}%",
                f"{single_blackhole - single_q8_0:+.1f} pts",
            )
        )
    if "multi-key" in niah_averages and Q8_0 in niah_averages["multi-key"] and BLACKHOLE_ALL in niah_averages["multi-key"]:
        multi_key_q8_0 = niah_averages["multi-key"][Q8_0]
        multi_key_blackhole = niah_averages["multi-key"][BLACKHOLE_ALL]
        metric_rows.append(
            (
                "NIAH multi-key expected accuracy",
                f"{multi_key_q8_0:.1f}%",
                f"{multi_key_blackhole:.1f}%",
                f"{multi_key_blackhole - multi_key_q8_0:+.1f} pts",
            )
        )
    if "multi-value" in niah_averages and Q8_0 in niah_averages["multi-value"] and BLACKHOLE_ALL in niah_averages["multi-value"]:
        multi_value_q8_0 = niah_averages["multi-value"][Q8_0]
        multi_value_blackhole = niah_averages["multi-value"][BLACKHOLE_ALL]
        metric_rows.append(
            (
                "NIAH multi-value expected accuracy",
                f"{multi_value_q8_0:.1f}%",
                f"{multi_value_blackhole:.1f}%",
                f"{multi_value_blackhole - multi_value_q8_0:+.1f} pts",
            )
        )

    metric_rows.extend(
        [
            (
                "KL drift vs f16",
                _format_metric(quality_metrics["q8_0_kld"], precision=4),
                _format_metric(quality_metrics["blackhole_kld"], precision=4),
                (
                    f"{quality_metrics['blackhole_kld'] - quality_metrics['q8_0_kld']:+.4f}"
                    if quality_metrics["q8_0_kld"] is not None and quality_metrics["blackhole_kld"] is not None
                    else "n/a"
                ),
            ),
            (
                "Long-context perplexity",
                _format_metric(quality_metrics["q8_0_ppl"]),
                _format_metric(quality_metrics["blackhole_ppl"]),
                (
                    f"{quality_metrics['blackhole_ppl'] - quality_metrics['q8_0_ppl']:+.2f}"
                    if quality_metrics["q8_0_ppl"] is not None and quality_metrics["blackhole_ppl"] is not None
                    else "n/a"
                ),
            ),
            (
                "Compression frontier vs q8_0",
                _format_metric(quality_metrics["q8_0_frontier"], suffix="x"),
                _format_metric(quality_metrics["blackhole_frontier"], suffix="x"),
                (
                    f"{quality_metrics['blackhole_frontier'] - quality_metrics['q8_0_frontier']:+.2f}x"
                    if quality_metrics["q8_0_frontier"] is not None
                    and quality_metrics["blackhole_frontier"] is not None
                    else "n/a"
                ),
            ),
        ]
    )

    artifact_rows = [
        (artifact.mode, _display_path(artifact.json_path), _display_path(artifact.markdown_path))
        for artifact in manifest.niah_artifacts
    ]

    parts = [
        "# Blackhole PoC — Executive Summary",
        "",
        f"> **Generated at (UTC):** {manifest.generated_at_utc} | **Seed:** {manifest.seed} | "
        f"**Baseline:** standard `{manifest.baseline}`",
        f"> **Quality source:** {quality_source_label}",
    ]
    if manifest.commit_sha:
        parts.append(f"> **Git commit:** `{manifest.commit_sha}`")
    if manifest.pytest_status:
        parts.append(f"> **Verification:** `pytest -q tests` -> **{manifest.pytest_status}**")
    parts.extend(
        [
            "",
            "## Headline",
            "",
            "Blackhole still reads best as a stack of additive improvements on top of standard `q8_0`: "
            "the routing and locality pillars move the speed and retrieval story, while the quality sections "
            f"now declare their source explicitly as {quality_source_label}.",
            "",
            "## Key Metrics — Full Blackhole vs q8_0",
            "",
            markdown_table(("Metric", "q8_0", "full Blackhole", "Delta"), metric_rows),
            "",
            "## Evidence Tiers",
            "",
            "- Prefill, decode, transport, and skip-rate sections are still `proxy` scenario-model outputs.",
            f"- Quality sections are currently sourced from {quality_source_label}.",
            "- NIAH summary numbers come from the latest generated artifact JSON for each mode.",
            "",
            "## Latest NIAH Artifacts",
            "",
        ]
    )
    if artifact_rows:
        parts.append(markdown_table(("Mode", "JSON", "Markdown"), artifact_rows))
    else:
        parts.append("No NIAH artifacts were found.")
    parts.extend(
        [
            "",
            "## Caveats",
            "",
            "- This is still a proof-of-concept suite, not a native Blackhole runtime benchmark.",
            "- Generated docs are now timestamped from one UTC manifest, but headline claims remain bounded by the highest evidence tier available per section.",
        ]
    )
    return "\n".join(parts).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs-dir",
        default=str(DOCS_DIR),
        help="Output directory for results.md and result_summary.md (default: %(default)s).",
    )
    parser.add_argument(
        "--niah-dir",
        default=str(NIAH_DIR),
        help="Directory containing generated NIAH artifacts (default: %(default)s).",
    )
    parser.add_argument(
        "--quality-source",
        choices=("proxy", "artifact"),
        default="proxy",
        help="Whether quality sections should render from proxy scripts or a measured artifact.",
    )
    parser.add_argument(
        "--quality-artifact",
        default=None,
        help="Measured-quality artifact JSON used when --quality-source artifact.",
    )
    parser.add_argument(
        "--pytest-status",
        default=None,
        help="Optional status string for the latest pytest run, for example '49 passed'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed label to stamp into generated docs (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline",
        default="q8_0",
        help="Baseline label to stamp into generated docs (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_absolute():
        docs_dir = (PROJECT_ROOT / docs_dir).resolve()
    niah_dir = Path(args.niah_dir).expanduser()
    if not niah_dir.is_absolute():
        niah_dir = (PROJECT_ROOT / niah_dir).resolve()
    quality_artifact = None
    if args.quality_artifact:
        quality_artifact = Path(args.quality_artifact).expanduser()
        if not quality_artifact.is_absolute():
            quality_artifact = (PROJECT_ROOT / quality_artifact).resolve()

    manifest = RunManifest(
        generated_at_utc=utc_now_iso(),
        baseline=args.baseline,
        seed=args.seed,
        pytest_status=args.pytest_status,
        quality_source=args.quality_source,
        quality_artifact=quality_artifact,
        commit_sha=_git_commit_sha(),
        niah_artifacts=discover_latest_niah_artifacts(niah_dir),
    )
    quality_source_label, quality_metrics = _quality_summary(args.quality_source, quality_artifact)

    docs_dir.mkdir(parents=True, exist_ok=True)
    results_path = docs_dir / "results.md"
    summary_path = docs_dir / "result_summary.md"
    results_path.write_text(_build_results_doc(manifest, quality_source_label))
    summary_path.write_text(_build_summary_doc(manifest, quality_source_label, quality_metrics))

    print(f"Generated: {results_path}")
    print(f"Generated: {summary_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
