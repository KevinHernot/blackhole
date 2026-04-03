#!/usr/bin/env python3
"""Generate a deterministic bundle-backed measured-quality fixture artifact.

The output is intentionally a synthetic harness-validation artifact, not a real
benchmark result. Its job is to exercise the measured-quality pipeline and give
the quality scripts an artifact-backed mode before the real offline/model eval
pipeline lands.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core.comparison_profiles import (  # noqa: E402
    ALLOWED_CONFIGURATION_LABELS,
    BLACKHOLE_ALL,
    F16,
    Q8_0,
    Q8_0_PORTAL_ATTENTION,
    Q8_0_PREDICTIVE_TRANSPORT,
    Q8_0_PROCEDURAL_WEIGHTS,
    Q8_0_SEMANTIC_PVS,
    Q8_0_TOKEN_MERGING,
)
from blackhole_core.evidence_tiers import ArtifactMetadata, EvidenceTier  # noqa: E402
from blackhole_core.measured_quality import (  # noqa: E402
    build_measured_quality_artifact_from_bundle_paths,
    bundle_serialized_bytes,
    save_context_eval,
    save_measured_quality_artifact,
)
from blackhole_core.real_model import TensorBundle, save_tensor_bundle  # noqa: E402

DEFAULT_OUTPUT = PROJECT_ROOT / "evals" / "artifacts" / "measured_quality_fixture.json"
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "evals" / "bundles" / "fixture"
DEFAULT_MANIFEST_OUTPUT = PROJECT_ROOT / "evals" / "artifacts" / "measured_quality_fixture_manifest.json"

NOISE_BY_CONFIGURATION = {
    F16: 0.0,
    Q8_0: 0.020,
    Q8_0_SEMANTIC_PVS: 0.016,
    Q8_0_PORTAL_ATTENTION: 0.017,
    Q8_0_PREDICTIVE_TRANSPORT: 0.018,
    Q8_0_PROCEDURAL_WEIGHTS: 0.012,
    Q8_0_TOKEN_MERGING: 0.022,
    BLACKHOLE_ALL: 0.024,
}

BYTE_SCALE_BY_CONFIGURATION = {
    F16: 1.00,
    Q8_0: 0.55,
    Q8_0_SEMANTIC_PVS: 0.55,
    Q8_0_PORTAL_ATTENTION: 0.55,
    Q8_0_PREDICTIVE_TRANSPORT: 0.55,
    Q8_0_PROCEDURAL_WEIGHTS: 0.48,
    Q8_0_TOKEN_MERGING: 0.50,
    BLACKHOLE_ALL: 0.42,
}


def _git_commit_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        return None


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "configuration"


def _reference_bundle(rng: np.random.Generator) -> TensorBundle:
    token_count = 48
    hidden_size = 16
    vocab_size = 64
    domains = tuple("docs" if index < 16 else "code" if index < 32 else "mixed" for index in range(token_count))
    return TensorBundle(
        k_cache=rng.normal(size=(token_count, hidden_size)),
        v_cache=rng.normal(size=(token_count, hidden_size)),
        activations=rng.normal(size=(3, hidden_size)),
        domains=domains,
        logits=rng.normal(size=(token_count, vocab_size)),
    )


def _candidate_bundle(
    reference_bundle: TensorBundle,
    rng: np.random.Generator,
    *,
    noise_scale: float,
) -> TensorBundle:
    if noise_scale == 0.0:
        return TensorBundle(
            k_cache=np.array(reference_bundle.k_cache, copy=True),
            v_cache=np.array(reference_bundle.v_cache, copy=True),
            activations=np.array(reference_bundle.activations, copy=True)
            if reference_bundle.activations is not None
            else None,
            domains=reference_bundle.domains,
            logits=np.array(reference_bundle.logits, copy=True)
            if reference_bundle.logits is not None
            else None,
        )

    return TensorBundle(
        k_cache=reference_bundle.k_cache + rng.normal(scale=noise_scale, size=reference_bundle.k_cache.shape),
        v_cache=reference_bundle.v_cache + rng.normal(scale=noise_scale, size=reference_bundle.v_cache.shape),
        activations=(
            reference_bundle.activations
            + rng.normal(scale=noise_scale, size=reference_bundle.activations.shape)
            if reference_bundle.activations is not None
            else None
        ),
        domains=reference_bundle.domains,
        logits=(
            reference_bundle.logits
            + rng.normal(scale=noise_scale, size=reference_bundle.logits.shape)
            if reference_bundle.logits is not None
            else None
        ),
    )


def _context_logits(
    rng: np.random.Generator,
    *,
    token_count: int,
    vocab_size: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    logits = rng.normal(size=(token_count, vocab_size))
    targets = np.asarray(rng.integers(0, vocab_size, size=token_count), dtype=int)
    if noise_scale:
        logits = logits + rng.normal(scale=noise_scale, size=logits.shape)
    return logits, targets


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Where to write the measured-quality fixture artifact JSON.",
    )
    parser.add_argument(
        "--bundle-dir",
        default=str(DEFAULT_BUNDLE_DIR),
        help="Directory where reference/candidate bundles and context files will be written.",
    )
    parser.add_argument(
        "--manifest-output",
        default=str(DEFAULT_MANIFEST_OUTPUT),
        help="Where to write the bundle-backed fixture manifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed for the synthetic harness fixture.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    bundle_dir = Path(args.bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    reference_bundle = _reference_bundle(rng)
    reference_bundle_path = bundle_dir / "bundle__f16_reference.npz"
    save_tensor_bundle(reference_bundle_path, reference_bundle)
    reference_bytes = bundle_serialized_bytes(reference_bundle)

    candidate_bundle_paths: dict[str, Path] = {}
    candidate_serialized_bytes: dict[str, int] = {}
    short_context_paths: dict[str, Path] = {}
    long_context_paths: dict[str, Path] = {}
    manifest_candidates: list[dict[str, str | int]] = []

    for configuration in ALLOWED_CONFIGURATION_LABELS:
        slug = _slugify(configuration)
        noise_scale = NOISE_BY_CONFIGURATION[configuration]
        candidate_bundle = _candidate_bundle(reference_bundle, rng, noise_scale=noise_scale)
        candidate_bundle_path = bundle_dir / f"bundle__{slug}.npz"
        save_tensor_bundle(candidate_bundle_path, candidate_bundle)

        short_logits, short_targets = _context_logits(
            rng,
            token_count=12,
            vocab_size=reference_bundle.logits.shape[1],
            noise_scale=noise_scale * 0.5,
        )
        long_logits, long_targets = _context_logits(
            rng,
            token_count=48,
            vocab_size=reference_bundle.logits.shape[1],
            noise_scale=noise_scale,
        )
        candidate_bytes = max(1, int(round(reference_bytes * BYTE_SCALE_BY_CONFIGURATION[configuration])))
        short_context_path = bundle_dir / f"context__{slug}__short.npz"
        long_context_path = bundle_dir / f"context__{slug}__long.npz"
        save_context_eval(short_context_path, short_logits, short_targets)
        save_context_eval(long_context_path, long_logits, long_targets)

        candidate_bundle_paths[configuration] = candidate_bundle_path
        candidate_serialized_bytes[configuration] = candidate_bytes
        short_context_paths[configuration] = short_context_path
        long_context_paths[configuration] = long_context_path
        manifest_candidates.append(
            {
                "configuration": configuration,
                "bundle_path": str(candidate_bundle_path),
                "candidate_serialized_bytes": candidate_bytes,
                "short_context_path": str(short_context_path),
                "long_context_path": str(long_context_path),
            }
        )

    artifact = build_measured_quality_artifact_from_bundle_paths(
        metadata=ArtifactMetadata(
            run_id="measured-quality-fixture",
            created_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            evidence_tier=EvidenceTier.MEASURED_OFFLINE,
            source_model="synthetic-fixture",
            corpus_name="synthetic-fixture",
            seed=args.seed,
            commit_sha=_git_commit_sha(),
            notes=(
                "Synthetic measured-quality harness fixture rebuilt from saved tensor bundles.",
                "Not a headline benchmark artifact.",
            ),
        ),
        reference_configuration=F16,
        reference_bundle_path=reference_bundle_path,
        candidate_bundle_paths=candidate_bundle_paths,
        candidate_serialized_bytes=candidate_serialized_bytes,
        short_context_paths=short_context_paths,
        long_context_paths=long_context_paths,
    )
    save_measured_quality_artifact(args.output, artifact)

    manifest_output = Path(args.manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(
            {
                "run_id": artifact.metadata.run_id,
                "created_at_utc": artifact.metadata.created_at_utc,
                "evidence_tier": artifact.metadata.evidence_tier.value,
                "source_model": artifact.metadata.source_model,
                "corpus_name": artifact.metadata.corpus_name,
                "seed": artifact.metadata.seed,
                "commit_sha": artifact.metadata.commit_sha,
                "notes": list(artifact.metadata.notes),
                "artifact_path": str(Path(args.output)),
                "reference_configuration": F16,
                "reference_bundle_path": str(reference_bundle_path),
                "reference_bytes": reference_bytes,
                "top_p_threshold": artifact.top_p_threshold,
                "bundle_dir": str(bundle_dir),
                "candidates": manifest_candidates,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved measured-quality fixture artifact to: {Path(args.output)}")
    print(f"Saved bundle-backed fixture manifest to: {manifest_output}")
    print(f"Saved fixture bundles to: {bundle_dir}")


if __name__ == "__main__":
    main()
