#!/usr/bin/env python3
"""Build a measured-quality artifact from a bundle manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core import (  # noqa: E402
    ArtifactMetadata,
    EvidenceTier,
    MeasuredQualityArtifact,
    add_frontier_vs_baseline,
    aggregate_measured_quality_metrics,
    build_measured_quality_artifact_from_bundle_paths,
    bundle_serialized_bytes,
    canonicalize_configuration,
    load_tensor_bundle,
    measure_quality_metrics_from_capture_paths,
    save_measured_quality_artifact,
)


def _resolve_manifest_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _load_manifest(path: str | Path) -> tuple[Path, dict[str, object]]:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    return manifest_path, payload


def _metadata_from_payload(manifest_path: Path, payload: dict[str, object]) -> ArtifactMetadata:
    notes = [str(note) for note in payload.get("notes", ())]
    reference_provenance = payload.get("reference_provenance")
    if isinstance(reference_provenance, dict):
        notes.append(
            "reference_provenance:"
            f"{reference_provenance.get('manifest_path')}|"
            f"{reference_provenance.get('source_model')}|"
            f"{reference_provenance.get('corpus_name')}|"
            f"seed={reference_provenance.get('seed')}"
        )
    candidate_provenance = payload.get("candidate_provenance")
    if isinstance(candidate_provenance, list):
        for candidate in candidate_provenance:
            if not isinstance(candidate, dict):
                continue
            notes.append(
                "candidate_provenance:"
                f"{candidate.get('configuration')}|"
                f"{candidate.get('backend')}|"
                f"{candidate.get('manifest_path')}"
            )
    return ArtifactMetadata(
        run_id=str(payload["run_id"]),
        created_at_utc=str(payload["created_at_utc"]),
        evidence_tier=EvidenceTier(str(payload.get("evidence_tier", EvidenceTier.MEASURED_OFFLINE.value))),
        source_model=str(payload.get("source_model", "unknown-source-model")),
        corpus_name=str(payload.get("corpus_name", manifest_path.stem)),
        seed=int(payload.get("seed", 0)),
        commit_sha=str(payload["commit_sha"]) if payload.get("commit_sha") is not None else None,
        notes=tuple(notes),
    )


def _build_from_sample_manifest(
    manifest_dir: Path,
    payload: dict[str, object],
    metadata: ArtifactMetadata,
) -> MeasuredQualityArtifact:
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit("Error: sample-based manifests must include a non-empty `samples` array.")

    metrics_by_configuration: dict[str, list] = {}
    reference_bytes_by_sample: list[int] = []

    for sample in samples:
        if not isinstance(sample, dict):
            raise SystemExit("Error: each sample entry must be an object.")
        reference_bundle_path = _resolve_manifest_path(manifest_dir, str(sample["reference_bundle_path"]))
        reference_bytes_by_sample.append(bundle_serialized_bytes(load_tensor_bundle(reference_bundle_path)))

        candidates = sample.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise SystemExit(
                f"Error: sample {sample.get('id', '<unknown>')} must include a non-empty `candidates` array."
            )

        for candidate in candidates:
            configuration = canonicalize_configuration(str(candidate["configuration"]))
            try:
                metrics = measure_quality_metrics_from_capture_paths(
                    reference_bundle_path,
                    _resolve_manifest_path(manifest_dir, str(candidate["bundle_path"])),
                    candidate_bundle_format=(
                        str(candidate["bundle_format"])
                        if candidate.get("bundle_format") is not None
                        else None
                    ),
                    candidate_captured_components=(
                        [str(component) for component in candidate.get("captured_components", ())]
                        if isinstance(candidate.get("captured_components"), list)
                        else None
                    ),
                    candidate_serialized_bytes=(
                        int(candidate["candidate_serialized_bytes"])
                        if candidate.get("candidate_serialized_bytes") is not None
                        else None
                    ),
                    top_p_threshold=float(payload.get("top_p_threshold", 0.90)),
                    short_context_path=_resolve_manifest_path(manifest_dir, candidate.get("short_context_path")),
                    long_context_path=_resolve_manifest_path(manifest_dir, candidate.get("long_context_path")),
                )
            except ValueError as exc:
                raise SystemExit(f"Error: {exc}") from exc
            metrics_by_configuration.setdefault(configuration, []).append(metrics)

    aggregated = {
        configuration: aggregate_measured_quality_metrics(metrics)
        for configuration, metrics in metrics_by_configuration.items()
    }
    return MeasuredQualityArtifact(
        metadata=metadata,
        reference_configuration=canonicalize_configuration(str(payload["reference_configuration"])),
        reference_bytes=int(round(sum(reference_bytes_by_sample) / len(reference_bytes_by_sample))),
        top_p_threshold=float(payload.get("top_p_threshold", 0.90)),
        measurements=add_frontier_vs_baseline(aggregated),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        help="Path to the captured-bundle manifest JSON.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path for the measured-quality artifact. Defaults to manifest artifact_path.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path, payload = _load_manifest(args.manifest)
    manifest_dir = manifest_path.parent

    output_path = _resolve_manifest_path(
        manifest_dir,
        args.output or payload.get("artifact_path"),
    )
    if output_path is None:
        raise SystemExit("Error: provide --output or artifact_path in the manifest.")

    metadata = _metadata_from_payload(manifest_path, payload)

    if payload.get("samples") is not None:
        artifact = _build_from_sample_manifest(manifest_dir, payload, metadata)
    else:
        candidate_bundle_paths = {}
        candidate_bundle_formats = {}
        candidate_captured_components = {}
        candidate_serialized_bytes = {}
        short_context_paths = {}
        long_context_paths = {}
        for candidate in payload.get("candidates", []):
            configuration = str(candidate["configuration"])
            candidate_bundle_paths[configuration] = _resolve_manifest_path(manifest_dir, str(candidate["bundle_path"]))
            if candidate.get("bundle_format") is not None:
                candidate_bundle_formats[configuration] = str(candidate["bundle_format"])
            if isinstance(candidate.get("captured_components"), list):
                candidate_captured_components[configuration] = [
                    str(component) for component in candidate["captured_components"]
                ]
            if candidate.get("candidate_serialized_bytes") is not None:
                candidate_serialized_bytes[configuration] = int(candidate["candidate_serialized_bytes"])
            short_path = _resolve_manifest_path(manifest_dir, candidate.get("short_context_path"))
            if short_path is not None:
                short_context_paths[configuration] = short_path
            long_path = _resolve_manifest_path(manifest_dir, candidate.get("long_context_path"))
            if long_path is not None:
                long_context_paths[configuration] = long_path

        try:
            artifact = build_measured_quality_artifact_from_bundle_paths(
                metadata=metadata,
                reference_configuration=str(payload["reference_configuration"]),
                reference_bundle_path=_resolve_manifest_path(manifest_dir, str(payload["reference_bundle_path"])),
                candidate_bundle_paths=candidate_bundle_paths,
                candidate_bundle_formats=candidate_bundle_formats or None,
                candidate_captured_components=candidate_captured_components or None,
                candidate_serialized_bytes=candidate_serialized_bytes or None,
                top_p_threshold=float(payload.get("top_p_threshold", 0.90)),
                short_context_paths=short_context_paths or None,
                long_context_paths=long_context_paths or None,
            )
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc
    save_measured_quality_artifact(output_path, artifact)
    print(f"Built measured-quality artifact: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
