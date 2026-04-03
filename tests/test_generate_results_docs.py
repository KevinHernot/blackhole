from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blackhole_core import (  # noqa: E402
    ArtifactMetadata,
    EvidenceTier,
    MeasuredQualityArtifact,
    MeasuredQualityMetrics,
    save_measured_quality_artifact,
)


def _load_script_module(module_name: str):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"blackhole_{module_name}_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"blackhole_{module_name}_tests"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_niah_artifact(root: Path, mode: str, timestamp: str, q8_0: float, blackhole: float) -> None:
    payload = {
        "mode": mode,
        "results": [
            {"configuration": "q8_0", "expected_accuracy_pct": q8_0},
            {"configuration": "blackhole (q8_0 + all 5)", "expected_accuracy_pct": blackhole},
        ],
    }
    (root / f"niah_{mode}_{timestamp}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / f"niah_{mode}_{timestamp}.md").write_text(
        f"# Mock {mode}\n\nExpected summary for {mode}.\n",
    )


def test_generate_results_docs_writes_both_docs_with_proxy_quality(tmp_path: Path):
    niah_dir = tmp_path / "niah"
    docs_dir = tmp_path / "docs"
    niah_dir.mkdir()

    _write_niah_artifact(niah_dir, "single", "20260331_010101", 80.7, 88.4)
    _write_niah_artifact(niah_dir, "multi-key", "20260331_010101", 76.6, 89.9)
    _write_niah_artifact(niah_dir, "multi-value", "20260331_010101", 72.0, 88.9)

    generator = _load_script_module("generate_results_docs")
    generator.main(
        [
            "--docs-dir",
            str(docs_dir),
            "--niah-dir",
            str(niah_dir),
            "--pytest-status",
            "49 passed",
        ]
    )

    results_path = docs_dir / "results.md"
    summary_path = docs_dir / "result_summary.md"
    assert results_path.exists()
    assert summary_path.exists()

    results_text = results_path.read_text()
    summary_text = summary_path.read_text()

    assert "Generated at (UTC):" in results_text
    assert "Quality source:** proxy scenario-model outputs" in results_text
    assert "niah_single_20260331_010101.json" in results_text
    assert "Generated With" in results_text

    assert "49 passed" in summary_text
    assert "NIAH single-needle expected accuracy" in summary_text
    assert "Evidence Tiers" in summary_text


def test_generate_results_docs_can_use_measured_quality_artifact(tmp_path: Path):
    niah_dir = tmp_path / "niah"
    docs_dir = tmp_path / "docs"
    artifact_path = tmp_path / "quality_artifact.json"
    niah_dir.mkdir()

    _write_niah_artifact(niah_dir, "single", "20260331_020202", 80.7, 88.4)
    _write_niah_artifact(niah_dir, "multi-key", "20260331_020202", 76.6, 89.9)
    _write_niah_artifact(niah_dir, "multi-value", "20260331_020202", 72.0, 88.9)

    save_measured_quality_artifact(
        artifact_path,
        MeasuredQualityArtifact(
            metadata=ArtifactMetadata(
                run_id="artifact-docs-test",
                created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                evidence_tier=EvidenceTier.MEASURED_OFFLINE,
                source_model="synthetic-test",
                corpus_name="unit-test",
                seed=42,
                notes=("generator test",),
            ),
            reference_configuration="f16",
            reference_bytes=8192,
            top_p_threshold=0.90,
            measurements={
                "q8_0": MeasuredQualityMetrics(
                    mean_kld=0.0190,
                    same_top_p_fraction=0.964,
                    mean_cosine=0.972,
                    mse=0.0114,
                    relative_l2=0.1200,
                    max_abs_error=0.3100,
                    serialized_bytes=2048,
                    compression_ratio=4.0,
                    short_context_perplexity=8.20,
                    long_context_perplexity=9.80,
                    stability_fraction=0.8367,
                    frontier_vs_baseline=1.00,
                ),
                "blackhole (q8_0 + all 5)": MeasuredQualityMetrics(
                    mean_kld=0.0120,
                    same_top_p_fraction=0.981,
                    mean_cosine=0.989,
                    mse=0.0072,
                    relative_l2=0.0840,
                    max_abs_error=0.2200,
                    serialized_bytes=1706,
                    compression_ratio=4.80,
                    short_context_perplexity=7.90,
                    long_context_perplexity=8.60,
                    stability_fraction=0.9186,
                    frontier_vs_baseline=1.24,
                ),
            },
        ),
    )

    generator = _load_script_module("generate_results_docs")
    generator.main(
        [
            "--docs-dir",
            str(docs_dir),
            "--niah-dir",
            str(niah_dir),
            "--quality-source",
            "artifact",
            "--quality-artifact",
            str(artifact_path),
        ]
    )

    results_text = (docs_dir / "results.md").read_text()
    summary_text = (docs_dir / "result_summary.md").read_text()

    assert "measured_offline artifact" in results_text
    assert "quality_artifact.json" in results_text
    assert "Compression frontier vs q8_0" in summary_text
    assert "measured_offline artifact from unit-test" in summary_text
