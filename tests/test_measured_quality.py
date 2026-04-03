from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blackhole_core import (  # noqa: E402
    ArtifactMetadata,
    BLACKHOLE_ALL,
    EvidenceTier,
    MeasuredQualityArtifact,
    MeasuredQualityMetrics,
    Q8_0,
    add_frontier_vs_baseline,
    build_measured_quality_artifact_from_bundle_paths,
    load_context_eval,
    load_measured_quality_artifact,
    measure_quality_metrics,
    measure_quality_metrics_from_bundle_paths,
    perplexity_from_logits,
    same_top_p_fraction,
    save_context_eval,
    save_measured_quality_artifact,
)
from blackhole_core.real_model import TensorBundle, save_tensor_bundle  # noqa: E402


def _load_script_module(module_name: str):
    module_path = SCRIPTS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"blackhole_{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"blackhole_{module_name}"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_python_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_from_rng(seed: int, *, token_count: int = 12, hidden_size: int = 8, vocab_size: int = 16) -> TensorBundle:
    rng = np.random.default_rng(seed)
    return TensorBundle(
        k_cache=rng.normal(size=(token_count, hidden_size)),
        v_cache=rng.normal(size=(token_count, hidden_size)),
        activations=rng.normal(size=(3, hidden_size)),
        domains=tuple("docs" for _ in range(token_count)),
        logits=rng.normal(size=(token_count, vocab_size)),
    )


def _quality_logits(seed: int, *, row_count: int, vocab_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(row_count, vocab_size)),
        rng.integers(low=0, high=vocab_size, size=row_count, endpoint=False),
    )


def _measured_metrics(reference: TensorBundle, candidate_seed: int, *, with_perplexity: bool) -> object:
    kwargs = {}
    if with_perplexity:
        short_logits, short_targets = _quality_logits(candidate_seed + 100, row_count=4)
        long_logits, long_targets = _quality_logits(candidate_seed + 200, row_count=6)
        kwargs = {
            "short_context_logits": short_logits,
            "short_context_targets": short_targets,
            "long_context_logits": long_logits,
            "long_context_targets": long_targets,
        }
    return measure_quality_metrics(reference, _bundle_from_rng(candidate_seed), **kwargs)


def test_same_top_p_fraction_is_one_for_identical_logits():
    logits = np.asarray([[3.0, 1.0, 0.0], [0.2, 0.1, -0.1]])
    assert same_top_p_fraction(logits, logits) == 1.0


def test_perplexity_from_logits_returns_positive_value():
    logits = np.asarray([[4.0, 1.0, 0.0], [0.1, 3.0, -1.0]])
    token_ids = np.asarray([0, 1], dtype=int)
    assert perplexity_from_logits(logits, token_ids) > 1.0


def test_measure_quality_metrics_returns_real_computed_fields():
    reference = _bundle_from_rng(1)
    candidate = _bundle_from_rng(2)
    short_logits = np.asarray([[1.0, 0.2, -0.3], [0.5, 2.0, -0.8]])
    short_targets = np.asarray([0, 1], dtype=int)
    long_logits = np.asarray([[0.2, 1.5, -0.4], [2.0, 0.1, -1.0], [0.4, -0.2, 1.2]])
    long_targets = np.asarray([1, 0, 2], dtype=int)

    metrics = measure_quality_metrics(
        reference,
        candidate,
        short_context_logits=short_logits,
        short_context_targets=short_targets,
        long_context_logits=long_logits,
        long_context_targets=long_targets,
    )

    assert metrics.mean_kld >= 0.0
    assert 0.0 <= metrics.same_top_p_fraction <= 1.0
    assert metrics.mean_cosine <= 1.0
    assert metrics.mse >= 0.0
    assert metrics.compression_ratio >= 1.0
    assert metrics.short_context_perplexity is not None
    assert metrics.long_context_perplexity is not None
    assert metrics.stability_fraction is not None


def test_measured_quality_artifact_round_trip_preserves_measurements(tmp_path: Path):
    reference = _bundle_from_rng(3)
    q8_0_candidate = _bundle_from_rng(4)
    blackhole_candidate = _bundle_from_rng(5)

    measurements = add_frontier_vs_baseline(
        {
            Q8_0: measure_quality_metrics(reference, q8_0_candidate),
            BLACKHOLE_ALL: measure_quality_metrics(reference, blackhole_candidate),
        }
    )
    artifact = MeasuredQualityArtifact(
        metadata=ArtifactMetadata(
            run_id="test-run",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_OFFLINE,
            source_model="synthetic-test",
            corpus_name="unit-test",
            seed=7,
            notes=("unit test",),
        ),
        reference_configuration="f16",
        reference_bytes=reference.k_cache.nbytes + reference.v_cache.nbytes + reference.logits.nbytes,
        top_p_threshold=0.90,
        measurements=measurements,
    )

    output_path = tmp_path / "artifact.json"
    save_measured_quality_artifact(output_path, artifact)
    loaded = load_measured_quality_artifact(output_path)

    assert loaded.metadata.evidence_tier == EvidenceTier.MEASURED_OFFLINE
    assert loaded.measurement_for(Q8_0).frontier_vs_baseline is not None
    assert loaded.measurement_for(BLACKHOLE_ALL).mean_kld >= 0.0


def test_bundle_backed_quality_metrics_match_saved_inputs(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "candidate.npz"
    short_context_path = tmp_path / "short_context.npz"
    long_context_path = tmp_path / "long_context.npz"

    reference = _bundle_from_rng(21)
    candidate = _bundle_from_rng(22)
    save_tensor_bundle(reference_path, reference)
    save_tensor_bundle(candidate_path, candidate)

    short_logits, short_targets = _quality_logits(121, row_count=4)
    long_logits, long_targets = _quality_logits(221, row_count=6)
    save_context_eval(short_context_path, short_logits, short_targets)
    save_context_eval(long_context_path, long_logits, long_targets)

    loaded_short_logits, loaded_short_targets = load_context_eval(short_context_path)
    assert loaded_short_logits.shape == short_logits.shape
    assert loaded_short_targets.shape == short_targets.shape

    metrics = measure_quality_metrics_from_bundle_paths(
        reference_path,
        candidate_path,
        short_context_path=short_context_path,
        long_context_path=long_context_path,
    )

    assert metrics.mean_kld >= 0.0
    assert metrics.short_context_perplexity is not None
    assert metrics.long_context_perplexity is not None
    assert metrics.stability_fraction is not None


def test_build_measured_quality_artifact_from_bundle_paths(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    q8_0_path = tmp_path / "q8_0.npz"
    blackhole_path = tmp_path / "blackhole.npz"
    save_tensor_bundle(reference_path, _bundle_from_rng(31))
    save_tensor_bundle(q8_0_path, _bundle_from_rng(32))
    save_tensor_bundle(blackhole_path, _bundle_from_rng(33))

    q8_0_short = tmp_path / "q8_0_short.npz"
    q8_0_long = tmp_path / "q8_0_long.npz"
    blackhole_short = tmp_path / "blackhole_short.npz"
    blackhole_long = tmp_path / "blackhole_long.npz"
    save_context_eval(q8_0_short, *_quality_logits(331, row_count=4))
    save_context_eval(q8_0_long, *_quality_logits(431, row_count=6))
    save_context_eval(blackhole_short, *_quality_logits(332, row_count=4))
    save_context_eval(blackhole_long, *_quality_logits(432, row_count=6))

    artifact = build_measured_quality_artifact_from_bundle_paths(
        metadata=ArtifactMetadata(
            run_id="bundle-backed",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_OFFLINE,
            source_model="synthetic-test",
            corpus_name="unit-test",
            seed=31,
            notes=("bundle backed",),
        ),
        reference_configuration="f16",
        reference_bundle_path=reference_path,
        candidate_bundle_paths={
            Q8_0: q8_0_path,
            BLACKHOLE_ALL: blackhole_path,
        },
        short_context_paths={
            Q8_0: q8_0_short,
            BLACKHOLE_ALL: blackhole_short,
        },
        long_context_paths={
            Q8_0: q8_0_long,
            BLACKHOLE_ALL: blackhole_long,
        },
    )

    assert artifact.reference_configuration == "f16"
    assert artifact.measurement_for(Q8_0).frontier_vs_baseline is not None
    assert artifact.measurement_for(BLACKHOLE_ALL).long_context_perplexity is not None


def test_quality_scripts_can_render_artifact_mode(tmp_path: Path):
    reference = _bundle_from_rng(6)
    measurements = add_frontier_vs_baseline(
        {
            Q8_0: _measured_metrics(reference, 7, with_perplexity=True),
            BLACKHOLE_ALL: _measured_metrics(reference, 8, with_perplexity=True),
        }
    )
    artifact = MeasuredQualityArtifact(
        metadata=ArtifactMetadata(
            run_id="render-test",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_OFFLINE,
            source_model="synthetic-test",
            corpus_name="unit-test",
            seed=9,
            notes=("render test",),
        ),
        reference_configuration="f16",
        reference_bytes=reference.k_cache.nbytes + reference.v_cache.nbytes + reference.logits.nbytes,
        top_p_threshold=0.90,
        measurements=measurements,
    )
    artifact_path = tmp_path / "artifact.json"
    save_measured_quality_artifact(artifact_path, artifact)

    for module_name, heading in (
        ("kl_divergence_vs_f16", "KL Divergence Proof of Concept"),
        ("long_context_perplexity", "Long-Context Perplexity Proof of Concept"),
        ("compression_quality", "Compression Quality Proof of Concept"),
    ):
        module = _load_script_module(module_name)
        stream = io.StringIO()
        with redirect_stdout(stream):
            module.main(["--source", "artifact", "--artifact", str(artifact_path)])
        output = stream.getvalue()
        assert heading in output
        assert "Artifact source:" in output
        assert "measured_offline artifact" in output


def test_long_context_artifact_mode_rejects_missing_perplexity_fields(tmp_path: Path):
    reference = _bundle_from_rng(11)
    artifact = MeasuredQualityArtifact(
        metadata=ArtifactMetadata(
            run_id="missing-perplexity",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_OFFLINE,
            source_model="synthetic-test",
            corpus_name="unit-test",
            seed=12,
            notes=("missing perplexity",),
        ),
        reference_configuration="f16",
        reference_bytes=reference.k_cache.nbytes + reference.v_cache.nbytes + reference.logits.nbytes,
        top_p_threshold=0.90,
        measurements=add_frontier_vs_baseline(
            {
                Q8_0: _measured_metrics(reference, 12, with_perplexity=False),
                BLACKHOLE_ALL: _measured_metrics(reference, 13, with_perplexity=False),
            }
        ),
    )
    artifact_path = tmp_path / "missing-perplexity.json"
    save_measured_quality_artifact(artifact_path, artifact)

    module = _load_script_module("long_context_perplexity")
    try:
        module.main(["--source", "artifact", "--artifact", str(artifact_path)])
    except SystemExit as exc:
        message = str(exc)
        assert "requires measured long-context perplexity" in message
        assert Q8_0 in message
    else:
        raise AssertionError("Expected long_context_perplexity artifact mode to reject incomplete artifacts")


def test_compression_quality_artifact_mode_rejects_runtime_observed_style_partial_artifact(tmp_path: Path):
    reference = _bundle_from_rng(14)
    artifact = MeasuredQualityArtifact(
        metadata=ArtifactMetadata(
            run_id="runtime-observed-partial",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_MODEL,
            source_model="synthetic-test",
            corpus_name="unit-test",
            seed=15,
            notes=("runtime observed partial",),
        ),
        reference_configuration="f16",
        reference_bytes=reference.k_cache.nbytes + reference.v_cache.nbytes + reference.logits.nbytes,
        top_p_threshold=0.90,
        measurements=add_frontier_vs_baseline(
            {
                Q8_0: MeasuredQualityMetrics(
                    mean_kld=0.011,
                    same_top_p_fraction=0.98,
                    mean_cosine=None,
                    mse=None,
                    relative_l2=None,
                    max_abs_error=None,
                    serialized_bytes=1536,
                    compression_ratio=None,
                    short_context_perplexity=8.1,
                    long_context_perplexity=9.4,
                    stability_fraction=8.1 / 9.4,
                ),
                BLACKHOLE_ALL: MeasuredQualityMetrics(
                    mean_kld=0.008,
                    same_top_p_fraction=0.99,
                    mean_cosine=None,
                    mse=None,
                    relative_l2=None,
                    max_abs_error=None,
                    serialized_bytes=1408,
                    compression_ratio=None,
                    short_context_perplexity=7.8,
                    long_context_perplexity=8.9,
                    stability_fraction=7.8 / 8.9,
                ),
            }
        ),
    )
    artifact_path = tmp_path / "runtime_observed_partial.json"
    save_measured_quality_artifact(artifact_path, artifact)

    module = _load_script_module("compression_quality")
    try:
        module.main(["--source", "artifact", "--artifact", str(artifact_path)])
    except SystemExit as exc:
        message = str(exc)
        assert "requires full tensor-bundle compression/reconstruction metrics" in message
        assert "runtime_observed_v1 captures support KL/perplexity promotion only" in message
    else:
        raise AssertionError("Expected compression_quality artifact mode to reject partial runtime-observed artifacts")


def test_fixture_generator_writes_bundle_backed_manifest(tmp_path: Path):
    artifact_path = tmp_path / "fixture.json"
    bundle_dir = tmp_path / "bundles"
    manifest_path = tmp_path / "fixture_manifest.json"
    module = _load_python_module(
        PROJECT_ROOT / "evals" / "generate_measured_quality_fixture.py",
        "blackhole_fixture_generator",
    )

    module.main(
        [
            "--output",
            str(artifact_path),
            "--bundle-dir",
            str(bundle_dir),
            "--manifest-output",
            str(manifest_path),
            "--seed",
            "17",
        ]
    )

    assert artifact_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_path"] == str(artifact_path)
    assert Path(manifest["reference_bundle_path"]).exists()
    assert manifest["candidates"]
    assert Path(manifest["candidates"][0]["bundle_path"]).exists()

    loaded = load_measured_quality_artifact(artifact_path)
    assert loaded.reference_configuration == "f16"
    assert loaded.measurement_for(Q8_0).short_context_perplexity is not None
