from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
EVALS_DIR = PROJECT_ROOT / "evals"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from blackhole_core import (  # noqa: E402
    BLACKHOLE_ALL,
    Q8_0,
    load_measured_quality_artifact,
    save_context_eval,
    save_tensor_bundle,
)
from blackhole_core.real_model import TensorBundle  # noqa: E402


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_from_seed(seed: int) -> TensorBundle:
    rng = np.random.default_rng(seed)
    return TensorBundle(
        k_cache=rng.normal(size=(8, 4)),
        v_cache=rng.normal(size=(8, 4)),
        activations=rng.normal(size=(3, 4)),
        domains=tuple("docs" for _ in range(8)),
        logits=rng.normal(size=(8, 7)),
    )


def _context_eval_from_seed(seed: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, 7)), rng.integers(0, 7, size=rows, endpoint=False)


def _write_reference_sample(tmp_path: Path, sample_id: str, seed: int) -> dict[str, str]:
    bundle_path = tmp_path / f"{sample_id}__reference.npz"
    short_path = tmp_path / f"{sample_id}__short.npz"
    long_path = tmp_path / f"{sample_id}__long.npz"
    save_tensor_bundle(bundle_path, _bundle_from_seed(seed))
    save_context_eval(short_path, *_context_eval_from_seed(seed + 100, 4))
    save_context_eval(long_path, *_context_eval_from_seed(seed + 200, 6))
    return {
        "id": sample_id,
        "prompt": f"prompt for {sample_id}",
        "reference_bundle_path": str(bundle_path),
        "short_context_path": str(short_path),
        "long_context_path": str(long_path),
        "token_count": 6,
        "short_context_tokens": 4,
    }


def _write_candidate_sample(tmp_path: Path, sample_id: str, seed: int) -> dict[str, str]:
    bundle_path = tmp_path / f"{sample_id}__candidate.npz"
    short_path = tmp_path / f"{sample_id}__candidate_short.npz"
    long_path = tmp_path / f"{sample_id}__candidate_long.npz"
    save_tensor_bundle(bundle_path, _bundle_from_seed(seed))
    save_context_eval(short_path, *_context_eval_from_seed(seed + 300, 4))
    save_context_eval(long_path, *_context_eval_from_seed(seed + 400, 6))
    return {
        "id": sample_id,
        "bundle_path": str(bundle_path),
        "short_context_path": str(short_path),
        "long_context_path": str(long_path),
        "token_count": 6,
        "short_context_tokens": 4,
        "candidate_serialized_bytes": 1024 + seed,
        "runtime_summary": {"decode_speedup": 1.0 + seed / 1000.0},
    }


def test_build_measured_quality_artifact_script_from_manifest(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    q8_0_path = tmp_path / "q8_0.npz"
    blackhole_path = tmp_path / "blackhole.npz"
    save_tensor_bundle(reference_path, _bundle_from_seed(1))
    save_tensor_bundle(q8_0_path, _bundle_from_seed(2))
    save_tensor_bundle(blackhole_path, _bundle_from_seed(3))

    q8_0_short = tmp_path / "q8_0_short.npz"
    q8_0_long = tmp_path / "q8_0_long.npz"
    blackhole_short = tmp_path / "blackhole_short.npz"
    blackhole_long = tmp_path / "blackhole_long.npz"
    save_context_eval(q8_0_short, *_context_eval_from_seed(10, 4))
    save_context_eval(q8_0_long, *_context_eval_from_seed(11, 6))
    save_context_eval(blackhole_short, *_context_eval_from_seed(12, 4))
    save_context_eval(blackhole_long, *_context_eval_from_seed(13, 6))

    manifest_path = tmp_path / "capture_manifest.json"
    artifact_path = tmp_path / "artifact.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "bundle-manifest-test",
                "created_at_utc": "2026-03-31T03:10:00Z",
                "evidence_tier": "measured_offline",
                "source_model": "synthetic-test",
                "corpus_name": "unit-test",
                "seed": 0,
                "artifact_path": str(artifact_path),
                "reference_configuration": "f16",
                "reference_bundle_path": str(reference_path),
                "candidates": [
                    {
                        "configuration": Q8_0,
                        "bundle_path": str(q8_0_path),
                        "short_context_path": str(q8_0_short),
                        "long_context_path": str(q8_0_long),
                    },
                    {
                        "configuration": BLACKHOLE_ALL,
                        "bundle_path": str(blackhole_path),
                        "short_context_path": str(blackhole_short),
                        "long_context_path": str(blackhole_long),
                    },
                ],
            },
            indent=2,
        )
        + "\n"
    )

    module = _load_module(EVALS_DIR / "build_measured_quality_artifact.py", "blackhole_build_measured_quality")
    module.main([str(manifest_path)])

    artifact = load_measured_quality_artifact(artifact_path)
    assert artifact.reference_configuration == "f16"
    assert artifact.measurement_for(Q8_0).short_context_perplexity is not None
    assert artifact.measurement_for(BLACKHOLE_ALL).long_context_perplexity is not None


def test_run_measured_model_eval_fails_cleanly_without_optional_deps(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text('{"id":"sample-1","prompt":"hello world prompt"}\n')
    module = _load_module(SCRIPTS_DIR / "run_measured_model_eval.py", "blackhole_run_measured_model_eval_missing")

    def _raise_import_error():
        raise RuntimeError("run_measured_model_eval.py requires optional bench dependencies: torch and transformers.")

    monkeypatch.setattr(module, "_import_bench_dependencies", _raise_import_error)
    try:
        module.main(["--model", "fake-model", "--corpus", str(corpus_path), "--output-dir", str(tmp_path / "out")])
    except SystemExit as exc:
        assert "requires optional bench dependencies" in str(exc)
    else:
        raise AssertionError("Expected measured-model eval to fail cleanly without optional deps")


def test_run_measured_model_eval_writes_reference_manifest_with_fakes(tmp_path: Path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id":"sample-1","prompt":"portal attention helps long prompts"}\n'
        '{"id":"sample-2","prompt":"predictive transport reduces bandwidth"}\n'
    )
    module = _load_module(SCRIPTS_DIR / "run_measured_model_eval.py", "blackhole_run_measured_model_eval_fake")

    class FakeTensor:
        def __init__(self, array):
            self.array = np.asarray(array)

        @property
        def shape(self):
            return self.array.shape

        def squeeze(self, axis=None):
            return FakeTensor(np.squeeze(self.array, axis=axis))

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.array

        def to(self, device):
            return self

        def __getitem__(self, key):
            return FakeTensor(self.array[key])

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        @staticmethod
        def no_grad():
            return FakeNoGrad()

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_name, trust_remote_code=False):
            return cls()

        def __call__(self, prompt, return_tensors="pt"):
            token_count = max(3, len(prompt.split()))
            token_ids = np.arange(1, token_count + 1, dtype=int)
            return {
                "input_ids": FakeTensor(token_ids.reshape(1, -1)),
                "attention_mask": FakeTensor(np.ones((1, token_count), dtype=int)),
            }

    class FakeOutputs:
        def __init__(self, seq_len: int):
            vocab_size = 9
            hidden_size = 5
            self.past_key_values = [
                (
                    FakeTensor(np.full((1, 1, seq_len, hidden_size), 0.1 * (layer + 1))),
                    FakeTensor(np.full((1, 1, seq_len, hidden_size), 0.2 * (layer + 1))),
                )
                for layer in range(2)
            ]
            self.hidden_states = [
                FakeTensor(np.full((1, seq_len, hidden_size), 0.3 * (index + 1)))
                for index in range(3)
            ]
            logits = np.tile(np.linspace(0.1, 0.9, vocab_size, dtype=float), (seq_len, 1))
            logits += np.arange(seq_len, dtype=float).reshape(-1, 1) * 0.01
            self.logits = FakeTensor(logits.reshape(1, seq_len, vocab_size))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name, trust_remote_code=False):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return None

        def __call__(self, **inputs):
            seq_len = inputs["input_ids"].array.shape[1]
            return FakeOutputs(seq_len)

    class FakeAutoTokenizer:
        from_pretrained = FakeTokenizer.from_pretrained

    class FakeAutoModel:
        from_pretrained = FakeModel.from_pretrained

    monkeypatch.setattr(
        module,
        "_import_bench_dependencies",
        lambda: (FakeTorch(), FakeAutoModel, FakeAutoTokenizer),
    )
    monkeypatch.setattr(module, "_git_commit_sha", lambda: "deadbeef")

    output_dir = tmp_path / "captured"
    module.main(
        [
            "--model",
            "fake-model",
            "--corpus",
            str(corpus_path),
            "--output-dir",
            str(output_dir),
            "--short-context-tokens",
            "4",
        ]
    )

    summary = json.loads((output_dir / "measured_model_eval.json").read_text())
    manifest = json.loads((output_dir / "captured_reference_manifest.json").read_text())

    assert summary["evidence_tier"] == "measured_model"
    assert summary["sample_count"] == 2
    assert summary["average_long_context_perplexity"] > 0.0
    assert manifest["reference_configuration"] == "f16"
    assert len(manifest["samples"]) == 2
    assert Path(manifest["samples"][0]["reference_bundle_path"]).exists()
    assert Path(manifest["samples"][0]["long_context_path"]).exists()


def test_merge_runtime_capture_manifests_and_build_artifact(tmp_path: Path):
    reference_manifest = tmp_path / "captured_reference_manifest.json"
    q8_0_manifest = tmp_path / "runtime_q8_0.json"
    blackhole_manifest = tmp_path / "runtime_blackhole.json"
    merged_manifest = tmp_path / "merged_runtime_manifest.json"
    artifact_path = tmp_path / "merged_artifact.json"

    reference_manifest.write_text(
        json.dumps(
            {
                "run_id": "reference-run",
                "created_at_utc": "2026-03-31T04:00:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "reference_configuration": "f16",
                "samples": [
                    _write_reference_sample(tmp_path, "sample-1", 41),
                    _write_reference_sample(tmp_path, "sample-2", 42),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    q8_0_manifest.write_text(
        json.dumps(
            {
                "run_id": "runtime-q8_0",
                "created_at_utc": "2026-03-31T04:01:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "configuration": Q8_0,
                "backend": "ggml",
                "backend_configuration": {"threads": 8, "gpu_layers": 0},
                "samples": [
                    _write_candidate_sample(tmp_path, "sample-1", 51),
                    _write_candidate_sample(tmp_path, "sample-2", 52),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    blackhole_manifest.write_text(
        json.dumps(
            {
                "run_id": "runtime-blackhole",
                "created_at_utc": "2026-03-31T04:02:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "configuration": BLACKHOLE_ALL,
                "backend": "ggml",
                "backend_configuration": {"threads": 8, "gpu_layers": 0},
                "samples": [
                    _write_candidate_sample(tmp_path, "sample-1", 61),
                    _write_candidate_sample(tmp_path, "sample-2", 62),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    merge_module = _load_module(EVALS_DIR / "merge_runtime_capture_manifests.py", "blackhole_merge_runtime_manifests")
    merge_module.main(
        [
            "--reference-manifest",
            str(reference_manifest),
            "--candidate-manifest",
            str(q8_0_manifest),
            "--candidate-manifest",
            str(blackhole_manifest),
            "--output",
            str(merged_manifest),
            "--artifact-path",
            str(artifact_path),
        ]
    )

    merged = json.loads(merged_manifest.read_text())
    assert merged["evidence_tier"] == "measured_model"
    assert len(merged["samples"]) == 2
    assert len(merged["samples"][0]["candidates"]) == 2
    assert merged["reference_provenance"]["source_model"] == "fake-model"
    assert merged["candidate_provenance"][0]["backend_configuration"]["threads"] == 8
    assert merged["samples"][0]["candidates"][0]["token_count"] == 6
    assert merged["samples"][0]["candidates"][0]["short_context_tokens"] == 4

    build_module = _load_module(EVALS_DIR / "build_measured_quality_artifact.py", "blackhole_build_measured_quality_merged")
    build_module.main([str(merged_manifest)])
    artifact = load_measured_quality_artifact(artifact_path)
    assert artifact.measurement_for(Q8_0).sample_count == 2
    assert artifact.measurement_for(BLACKHOLE_ALL).sample_count == 2
    assert any("candidate_provenance:q8_0|ggml|" in note for note in artifact.metadata.notes)


def test_merge_runtime_capture_manifests_rejects_missing_candidate_samples(tmp_path: Path):
    reference_manifest = tmp_path / "captured_reference_manifest.json"
    bad_candidate_manifest = tmp_path / "runtime_bad.json"
    merged_manifest = tmp_path / "merged_runtime_manifest.json"

    reference_manifest.write_text(
        json.dumps(
            {
                "run_id": "reference-run",
                "created_at_utc": "2026-03-31T04:10:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "reference_configuration": "f16",
                "samples": [
                    _write_reference_sample(tmp_path, "sample-1", 71),
                    _write_reference_sample(tmp_path, "sample-2", 72),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    bad_candidate_manifest.write_text(
        json.dumps(
            {
                "run_id": "runtime-q8_0",
                "created_at_utc": "2026-03-31T04:11:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "configuration": Q8_0,
                "backend": "ggml",
                "backend_configuration": {"threads": 4},
                "samples": [
                    _write_candidate_sample(tmp_path, "sample-1", 81),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    merge_module = _load_module(EVALS_DIR / "merge_runtime_capture_manifests.py", "blackhole_merge_runtime_manifests_bad")
    try:
        merge_module.main(
            [
                "--reference-manifest",
                str(reference_manifest),
                "--candidate-manifest",
                str(bad_candidate_manifest),
                "--output",
                str(merged_manifest),
            ]
        )
    except SystemExit as exc:
        assert "missing sample ids" in str(exc)
    else:
        raise AssertionError("Expected runtime manifest merge to reject missing sample ids")


def test_merge_runtime_capture_manifests_rejects_missing_backend_configuration(tmp_path: Path):
    reference_manifest = tmp_path / "captured_reference_manifest.json"
    bad_candidate_manifest = tmp_path / "runtime_bad.json"
    merged_manifest = tmp_path / "merged_runtime_manifest.json"

    reference_manifest.write_text(
        json.dumps(
            {
                "run_id": "reference-run",
                "created_at_utc": "2026-03-31T04:20:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "reference_configuration": "f16",
                "samples": [
                    _write_reference_sample(tmp_path, "sample-1", 111),
                ],
            },
            indent=2,
        )
        + "\n"
    )
    bad_candidate_manifest.write_text(
        json.dumps(
            {
                "run_id": "runtime-q8_0",
                "created_at_utc": "2026-03-31T04:21:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "configuration": Q8_0,
                "backend": "ggml",
                "samples": [
                    _write_candidate_sample(tmp_path, "sample-1", 121),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    merge_module = _load_module(EVALS_DIR / "merge_runtime_capture_manifests.py", "blackhole_merge_runtime_manifests_missing_backend")
    try:
        merge_module.main(
            [
                "--reference-manifest",
                str(reference_manifest),
                "--candidate-manifest",
                str(bad_candidate_manifest),
                "--output",
                str(merged_manifest),
            ]
        )
    except SystemExit as exc:
        assert "backend_configuration" in str(exc)
    else:
        raise AssertionError("Expected runtime manifest merge to require backend_configuration")


def test_init_runtime_candidate_manifest_builds_runtime_template(tmp_path: Path):
    reference_manifest = tmp_path / "captured_reference_manifest.json"
    candidate_manifest = tmp_path / "runtime_q8_0.json"
    capture_root = tmp_path / "runtime_capture"

    reference_manifest.write_text(
        json.dumps(
            {
                "run_id": "reference-run",
                "created_at_utc": "2026-03-31T05:00:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 7,
                "reference_configuration": "f16",
                "samples": [
                    _write_reference_sample(tmp_path, "sample-1", 91),
                    _write_reference_sample(tmp_path, "sample-2", 92),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    init_module = _load_module(EVALS_DIR / "init_runtime_candidate_manifest.py", "blackhole_init_runtime_candidate_manifest")
    init_module.main(
        [
            "--reference-manifest",
            str(reference_manifest),
            "--configuration",
            Q8_0,
            "--backend",
            "ggml",
            "--backend-config-json",
            '{"threads": 12, "gpu_layers": 0}',
            "--capture-root",
            str(capture_root),
            "--output",
            str(candidate_manifest),
            "--commit-sha",
            "cafebabe",
        ]
    )

    payload = json.loads(candidate_manifest.read_text())
    assert payload["source_model"] == "fake-model"
    assert payload["corpus_name"] == "unit-test"
    assert payload["seed"] == 7
    assert payload["configuration"] == Q8_0
    assert payload["backend_configuration"]["threads"] == 12
    assert payload["samples"][0]["prompt"] == "prompt for sample-1"
    assert payload["samples"][0]["bundle_path"].endswith("runtime_capture/bundles/sample-1__q8_0.npz")
    assert payload["samples"][0]["short_context_path"].endswith("runtime_capture/contexts/sample-1__q8_0__short.npz")
    assert payload["samples"][0]["token_count"] == 6
    assert payload["samples"][0]["short_context_tokens"] == 4


def test_build_measured_quality_artifact_accepts_runtime_observed_bundle_format(tmp_path: Path):
    reference_path = tmp_path / "reference.npz"
    candidate_path = tmp_path / "runtime_observed_candidate.npz"
    artifact_path = tmp_path / "artifact.json"
    short_context_path = tmp_path / "runtime_observed_short.npz"
    long_context_path = tmp_path / "runtime_observed_long.npz"
    reference_bundle = _bundle_from_seed(130)
    save_tensor_bundle(reference_path, reference_bundle)
    save_context_eval(short_context_path, *_context_eval_from_seed(230, 4))
    save_context_eval(long_context_path, *_context_eval_from_seed(330, 6))
    np.savez(
        candidate_path,
        token_ids=np.arange(reference_bundle.logits.shape[0], dtype=np.int32),
        token_embeddings=np.asarray(reference_bundle.k_cache + 0.01, dtype=np.float32),
        logits=np.asarray(reference_bundle.logits + 0.02, dtype=np.float32),
    )

    manifest_path = tmp_path / "observed_capture_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "observed-runtime-capture-test",
                "created_at_utc": "2026-03-31T05:30:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "artifact_path": str(artifact_path),
                "reference_configuration": "f16",
                "samples": [
                    {
                        "id": "sample-1",
                        "prompt": "prompt for sample-1",
                        "reference_bundle_path": str(reference_path),
                        "candidates": [
                            {
                                "configuration": Q8_0,
                                "bundle_path": str(candidate_path),
                                "short_context_path": str(short_context_path),
                                "long_context_path": str(long_context_path),
                                "bundle_format": "runtime_observed_v1",
                                "captured_components": ["token_embeddings", "logits"],
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )

    module = _load_module(
        EVALS_DIR / "build_measured_quality_artifact.py",
        "blackhole_build_measured_quality_runtime_observed_accept",
    )
    module.main([str(manifest_path)])

    artifact = load_measured_quality_artifact(artifact_path)
    metrics = artifact.measurement_for(Q8_0)
    assert metrics.sample_count == 1
    assert metrics.mean_kld is not None
    assert metrics.same_top_p_fraction is not None
    assert metrics.short_context_perplexity is not None
    assert metrics.long_context_perplexity is not None
    assert metrics.compression_ratio is None
    assert metrics.mean_cosine is None
    assert metrics.frontier_vs_baseline is None


def test_build_measured_quality_artifact_script_accepts_runtime_observed_bundle_format_in_flat_manifest(tmp_path: Path):
    reference_path = tmp_path / "reference_flat.npz"
    candidate_path = tmp_path / "runtime_observed_candidate_flat.npz"
    artifact_path = tmp_path / "artifact_flat.json"
    short_context_path = tmp_path / "runtime_observed_flat_short.npz"
    long_context_path = tmp_path / "runtime_observed_flat_long.npz"
    reference_bundle = _bundle_from_seed(140)
    save_tensor_bundle(reference_path, reference_bundle)
    save_context_eval(short_context_path, *_context_eval_from_seed(240, 4))
    save_context_eval(long_context_path, *_context_eval_from_seed(340, 6))
    np.savez(
        candidate_path,
        token_ids=np.arange(reference_bundle.logits.shape[0], dtype=np.int32),
        token_embeddings=np.asarray(reference_bundle.k_cache + 0.015, dtype=np.float32),
        logits=np.asarray(reference_bundle.logits + 0.025, dtype=np.float32),
    )

    manifest_path = tmp_path / "observed_capture_flat_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "observed-runtime-capture-flat-test",
                "created_at_utc": "2026-03-31T05:31:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "artifact_path": str(artifact_path),
                "reference_configuration": "f16",
                "reference_bundle_path": str(reference_path),
                "candidates": [
                    {
                        "configuration": Q8_0,
                        "bundle_path": str(candidate_path),
                        "short_context_path": str(short_context_path),
                        "long_context_path": str(long_context_path),
                        "bundle_format": "runtime_observed_v1",
                        "captured_components": ["token_embeddings", "logits"],
                    }
                ],
            },
            indent=2,
        )
        + "\n"
    )

    module = _load_module(
        EVALS_DIR / "build_measured_quality_artifact.py",
        "blackhole_build_measured_quality_runtime_observed_flat_accept",
    )
    module.main([str(manifest_path)])

    artifact = load_measured_quality_artifact(artifact_path)
    metrics = artifact.measurement_for(Q8_0)
    assert metrics.mean_kld is not None
    assert metrics.long_context_perplexity is not None
    assert metrics.compression_ratio is None
    assert metrics.frontier_vs_baseline is None


def test_init_runtime_candidate_manifest_requires_backend_configuration(tmp_path: Path):
    reference_manifest = tmp_path / "captured_reference_manifest.json"
    candidate_manifest = tmp_path / "runtime_q8_0.json"

    reference_manifest.write_text(
        json.dumps(
            {
                "run_id": "reference-run",
                "created_at_utc": "2026-03-31T05:10:00Z",
                "evidence_tier": "measured_model",
                "source_model": "fake-model",
                "corpus_name": "unit-test",
                "seed": 0,
                "reference_configuration": "f16",
                "samples": [
                    _write_reference_sample(tmp_path, "sample-1", 101),
                ],
            },
            indent=2,
        )
        + "\n"
    )

    init_module = _load_module(EVALS_DIR / "init_runtime_candidate_manifest.py", "blackhole_init_runtime_candidate_manifest_bad")
    try:
        init_module.main(
            [
                "--reference-manifest",
                str(reference_manifest),
                "--configuration",
                Q8_0,
                "--backend",
                "ggml",
                "--capture-root",
                str(tmp_path / "runtime_capture"),
                "--output",
                str(candidate_manifest),
            ]
        )
    except SystemExit as exc:
        assert "exactly one of --backend-config-json or --backend-config-file" in str(exc)
    else:
        raise AssertionError("Expected runtime candidate init to require backend configuration")
