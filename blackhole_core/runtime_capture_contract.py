from __future__ import annotations

"""Validated contracts for measured runtime capture manifests.

This module defines the seam between the algorithm/eval repo and the sibling
``blackhole_runtime`` project. The runtime project is expected to emit
candidate-side capture manifests that satisfy this contract; this repo then
merges, validates, and summarizes those artifacts without re-implementing the
runtime backend.
"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .comparison_profiles import canonicalize_configuration
from .evidence_tiers import EvidenceTier

_MIN_CAPTURE_TIER = EvidenceTier.MEASURED_MODEL


def _load_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    return resolved, json.loads(resolved.read_text())


def _require_object(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_string(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must define non-empty `{key}`.")
    return value


def _optional_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"`{key}` must be a string when present.")
    stripped = value.strip()
    return stripped or None


def _require_integer(payload: Mapping[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{context} must define integer `{key}`.")
    return value


def _optional_integer(payload: Mapping[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"`{key}` must be an integer when present.")
    return value


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        raise ValueError(f"`{key}` must be numeric when present.")
    return float(value)


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    context: str,
    allow_empty: bool = True,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must define object `{key}`.")
    if not allow_empty and not value:
        raise ValueError(f"{context} must define non-empty `{key}`.")
    return value


def _require_utc_timestamp(value: str, *, context: str) -> str:
    if not value.endswith("Z"):
        raise ValueError(f"{context} must end with `Z` to indicate UTC.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{context} must be ISO-8601 UTC, got {value!r}.") from exc
    return value


def _require_capture_tier(value: str, *, context: str) -> EvidenceTier:
    tier = EvidenceTier(value)
    if tier.rank < _MIN_CAPTURE_TIER.rank:
        raise ValueError(
            f"{context} must be at least `{_MIN_CAPTURE_TIER.value}`, got `{tier.value}`."
        )
    return tier


def _resolve_path(base_dir: Path, raw: str | None, *, field_name: str) -> Path | None:
    if raw is None:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _configuration_slug(configuration: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", canonicalize_configuration(configuration).lower()).strip("_")
    return slug or "configuration"


@dataclass(frozen=True)
class ReferenceCaptureSample:
    id: str
    prompt: str
    reference_bundle_path: Path
    short_context_path: Path | None = None
    long_context_path: Path | None = None
    token_count: int | None = None
    short_context_tokens: int | None = None
    short_context_perplexity: float | None = None
    long_context_perplexity: float | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: Path, context: str) -> "ReferenceCaptureSample":
        return cls(
            id=_require_string(payload, "id", context=context),
            prompt=_require_string(payload, "prompt", context=context),
            reference_bundle_path=_resolve_path(
                base_dir,
                _require_string(payload, "reference_bundle_path", context=context),
                field_name="reference_bundle_path",
            ),
            short_context_path=_resolve_path(base_dir, _optional_string(payload, "short_context_path"), field_name="short_context_path"),
            long_context_path=_resolve_path(base_dir, _optional_string(payload, "long_context_path"), field_name="long_context_path"),
            token_count=_optional_integer(payload, "token_count"),
            short_context_tokens=_optional_integer(payload, "short_context_tokens"),
            short_context_perplexity=_optional_float(payload, "short_context_perplexity"),
            long_context_perplexity=_optional_float(payload, "long_context_perplexity"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "reference_bundle_path": str(self.reference_bundle_path),
        }
        if self.short_context_path is not None:
            payload["short_context_path"] = str(self.short_context_path)
        if self.long_context_path is not None:
            payload["long_context_path"] = str(self.long_context_path)
        if self.token_count is not None:
            payload["token_count"] = self.token_count
        if self.short_context_tokens is not None:
            payload["short_context_tokens"] = self.short_context_tokens
        if self.short_context_perplexity is not None:
            payload["short_context_perplexity"] = self.short_context_perplexity
        if self.long_context_perplexity is not None:
            payload["long_context_perplexity"] = self.long_context_perplexity
        return payload


@dataclass(frozen=True)
class RuntimeCandidateSample:
    id: str
    bundle_path: Path
    prompt: str | None = None
    short_context_path: Path | None = None
    long_context_path: Path | None = None
    token_count: int | None = None
    short_context_tokens: int | None = None
    candidate_serialized_bytes: int | None = None
    bundle_format: str | None = None
    captured_components: tuple[str, ...] = ()
    runtime_summary: Mapping[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, base_dir: Path, context: str) -> "RuntimeCandidateSample":
        runtime_summary = payload.get("runtime_summary")
        if runtime_summary is not None and not isinstance(runtime_summary, Mapping):
            raise ValueError(f"{context} `runtime_summary` must be an object when present.")
        captured_components_raw = payload.get("captured_components")
        if captured_components_raw is None:
            captured_components: tuple[str, ...] = ()
        else:
            if not isinstance(captured_components_raw, list):
                raise ValueError(f"{context} `captured_components` must be an array when present.")
            captured_components = tuple(str(component) for component in captured_components_raw)
        return cls(
            id=_require_string(payload, "id", context=context),
            prompt=_optional_string(payload, "prompt"),
            bundle_path=_resolve_path(
                base_dir,
                _require_string(payload, "bundle_path", context=context),
                field_name="bundle_path",
            ),
            short_context_path=_resolve_path(base_dir, _optional_string(payload, "short_context_path"), field_name="short_context_path"),
            long_context_path=_resolve_path(base_dir, _optional_string(payload, "long_context_path"), field_name="long_context_path"),
            token_count=_optional_integer(payload, "token_count"),
            short_context_tokens=_optional_integer(payload, "short_context_tokens"),
            candidate_serialized_bytes=_optional_integer(payload, "candidate_serialized_bytes"),
            bundle_format=_optional_string(payload, "bundle_format"),
            captured_components=captured_components,
            runtime_summary=dict(runtime_summary) if runtime_summary is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "bundle_path": str(self.bundle_path),
        }
        if self.prompt is not None:
            payload["prompt"] = self.prompt
        if self.short_context_path is not None:
            payload["short_context_path"] = str(self.short_context_path)
        if self.long_context_path is not None:
            payload["long_context_path"] = str(self.long_context_path)
        if self.token_count is not None:
            payload["token_count"] = self.token_count
        if self.short_context_tokens is not None:
            payload["short_context_tokens"] = self.short_context_tokens
        if self.candidate_serialized_bytes is not None:
            payload["candidate_serialized_bytes"] = self.candidate_serialized_bytes
        if self.bundle_format is not None:
            payload["bundle_format"] = self.bundle_format
        if self.captured_components:
            payload["captured_components"] = list(self.captured_components)
        if self.runtime_summary is not None:
            payload["runtime_summary"] = dict(self.runtime_summary)
        return payload


def _ensure_unique_sample_ids(samples: Sequence[ReferenceCaptureSample | RuntimeCandidateSample], *, context: str) -> None:
    ids = [sample.id for sample in samples]
    duplicates = sorted(sample_id for sample_id in set(ids) if ids.count(sample_id) > 1)
    if duplicates:
        raise ValueError(f"{context} contains duplicate sample ids: {', '.join(duplicates)}.")


@dataclass(frozen=True)
class ReferenceCaptureManifest:
    manifest_path: Path
    run_id: str
    created_at_utc: str
    evidence_tier: EvidenceTier
    source_model: str
    corpus_name: str
    seed: int
    reference_configuration: str
    samples: tuple[ReferenceCaptureSample, ...]
    commit_sha: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_path(cls, path: str | Path) -> "ReferenceCaptureManifest":
        manifest_path, payload = _load_json(path)
        return cls.from_dict(payload, manifest_path=manifest_path)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        manifest_path: str | Path,
    ) -> "ReferenceCaptureManifest":
        manifest_file = Path(manifest_path).expanduser().resolve()
        context = f"reference manifest {manifest_file}"
        samples_payload = payload.get("samples")
        if not isinstance(samples_payload, list) or not samples_payload:
            raise ValueError(f"{context} must define a non-empty `samples` array.")
        samples = tuple(
            ReferenceCaptureSample.from_dict(
                _require_object(sample, context=f"{context} sample"),
                base_dir=manifest_file.parent,
                context=f"{context} sample",
            )
            for sample in samples_payload
        )
        _ensure_unique_sample_ids(samples, context=context)
        created_at_utc = _require_utc_timestamp(
            _require_string(payload, "created_at_utc", context=context),
            context=f"{context} `created_at_utc`",
        )
        return cls(
            manifest_path=manifest_file,
            run_id=_require_string(payload, "run_id", context=context),
            created_at_utc=created_at_utc,
            evidence_tier=_require_capture_tier(
                _require_string(payload, "evidence_tier", context=context),
                context=f"{context} `evidence_tier`",
            ),
            source_model=_require_string(payload, "source_model", context=context),
            corpus_name=_require_string(payload, "corpus_name", context=context),
            seed=_require_integer(payload, "seed", context=context),
            reference_configuration=canonicalize_configuration(
                _require_string(payload, "reference_configuration", context=context)
            ),
            samples=samples,
            commit_sha=_optional_string(payload, "commit_sha"),
            notes=tuple(str(note) for note in payload.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "evidence_tier": self.evidence_tier.value,
            "source_model": self.source_model,
            "corpus_name": self.corpus_name,
            "seed": self.seed,
            "commit_sha": self.commit_sha,
            "notes": list(self.notes),
            "reference_configuration": self.reference_configuration,
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.id for sample in self.samples)


@dataclass(frozen=True)
class RuntimeCandidateManifest:
    manifest_path: Path
    run_id: str
    created_at_utc: str
    evidence_tier: EvidenceTier
    source_model: str
    corpus_name: str
    seed: int
    configuration: str
    backend: str
    backend_configuration: Mapping[str, Any]
    samples: tuple[RuntimeCandidateSample, ...]
    commit_sha: str | None = None
    notes: tuple[str, ...] = ()

    @classmethod
    def from_path(cls, path: str | Path) -> "RuntimeCandidateManifest":
        manifest_path, payload = _load_json(path)
        return cls.from_dict(payload, manifest_path=manifest_path)

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        manifest_path: str | Path,
    ) -> "RuntimeCandidateManifest":
        manifest_file = Path(manifest_path).expanduser().resolve()
        context = f"candidate manifest {manifest_file}"
        samples_payload = payload.get("samples")
        if not isinstance(samples_payload, list) or not samples_payload:
            raise ValueError(f"{context} must define a non-empty `samples` array.")
        samples = tuple(
            RuntimeCandidateSample.from_dict(
                _require_object(sample, context=f"{context} sample"),
                base_dir=manifest_file.parent,
                context=f"{context} sample",
            )
            for sample in samples_payload
        )
        _ensure_unique_sample_ids(samples, context=context)
        created_at_utc = _require_utc_timestamp(
            _require_string(payload, "created_at_utc", context=context),
            context=f"{context} `created_at_utc`",
        )
        return cls(
            manifest_path=manifest_file,
            run_id=_require_string(payload, "run_id", context=context),
            created_at_utc=created_at_utc,
            evidence_tier=_require_capture_tier(
                _require_string(payload, "evidence_tier", context=context),
                context=f"{context} `evidence_tier`",
            ),
            source_model=_require_string(payload, "source_model", context=context),
            corpus_name=_require_string(payload, "corpus_name", context=context),
            seed=_require_integer(payload, "seed", context=context),
            configuration=canonicalize_configuration(_require_string(payload, "configuration", context=context)),
            backend=_require_string(payload, "backend", context=context),
            backend_configuration=dict(
                _require_mapping(
                    payload,
                    "backend_configuration",
                    context=context,
                    allow_empty=False,
                )
            ),
            samples=samples,
            commit_sha=_optional_string(payload, "commit_sha"),
            notes=tuple(str(note) for note in payload.get("notes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "evidence_tier": self.evidence_tier.value,
            "source_model": self.source_model,
            "corpus_name": self.corpus_name,
            "seed": self.seed,
            "commit_sha": self.commit_sha,
            "notes": list(self.notes),
            "configuration": self.configuration,
            "backend": self.backend,
            "backend_configuration": dict(self.backend_configuration),
            "samples": [sample.to_dict() for sample in self.samples],
        }

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.id for sample in self.samples)

    def provenance_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "evidence_tier": self.evidence_tier.value,
            "source_model": self.source_model,
            "corpus_name": self.corpus_name,
            "seed": self.seed,
            "commit_sha": self.commit_sha,
            "configuration": self.configuration,
            "backend": self.backend,
            "backend_configuration": dict(self.backend_configuration),
        }


def build_runtime_candidate_manifest_template(
    reference_manifest: ReferenceCaptureManifest,
    *,
    configuration: str,
    backend: str,
    backend_configuration: Mapping[str, Any],
    capture_root: str | Path,
    run_id: str | None = None,
    created_at_utc: str,
    evidence_tier: EvidenceTier = EvidenceTier.MEASURED_MODEL,
    commit_sha: str | None = None,
    notes: Sequence[str] = (),
) -> RuntimeCandidateManifest:
    if evidence_tier.rank < _MIN_CAPTURE_TIER.rank:
        raise ValueError(
            f"candidate template evidence tier must be at least `{_MIN_CAPTURE_TIER.value}`, got `{evidence_tier.value}`."
        )
    if not backend_configuration:
        raise ValueError("candidate template requires non-empty backend_configuration.")

    configuration_label = canonicalize_configuration(configuration)
    capture_root_path = Path(capture_root).expanduser().resolve()
    slug = _configuration_slug(configuration_label)
    samples = tuple(
        RuntimeCandidateSample(
            id=sample.id,
            prompt=sample.prompt,
            bundle_path=capture_root_path / "bundles" / f"{sample.id}__{slug}.npz",
            short_context_path=capture_root_path / "contexts" / f"{sample.id}__{slug}__short.npz",
            long_context_path=capture_root_path / "contexts" / f"{sample.id}__{slug}__long.npz",
            token_count=sample.token_count,
            short_context_tokens=sample.short_context_tokens,
            runtime_summary={},
        )
        for sample in reference_manifest.samples
    )
    return RuntimeCandidateManifest(
        manifest_path=capture_root_path / f"{slug}_runtime_candidate_manifest.json",
        run_id=run_id or f"{reference_manifest.run_id}-{slug}-runtime-capture",
        created_at_utc=_require_utc_timestamp(created_at_utc, context="candidate template `created_at_utc`"),
        evidence_tier=evidence_tier,
        source_model=reference_manifest.source_model,
        corpus_name=reference_manifest.corpus_name,
        seed=reference_manifest.seed,
        configuration=configuration_label,
        backend=backend.strip(),
        backend_configuration=dict(backend_configuration),
        samples=samples,
        commit_sha=commit_sha,
        notes=tuple(notes),
    )


def merge_runtime_capture_manifests(
    reference_manifest: ReferenceCaptureManifest,
    candidate_manifests: Sequence[RuntimeCandidateManifest],
    *,
    created_at_utc: str,
    output_path: str | Path,
    artifact_path: str | Path | None = None,
    top_p_threshold: float = 0.90,
) -> dict[str, Any]:
    if not candidate_manifests:
        raise ValueError("at least one candidate manifest is required.")
    if top_p_threshold <= 0.0 or top_p_threshold > 1.0:
        raise ValueError(f"top_p_threshold must be in (0, 1], got {top_p_threshold}.")

    output_manifest_path = Path(output_path).expanduser().resolve()
    artifact_manifest_path = Path(artifact_path).expanduser().resolve() if artifact_path is not None else None

    seen_configurations: set[str] = set()
    merged_samples = {
        sample.id: {
            "id": sample.id,
            "prompt": sample.prompt,
            "reference_bundle_path": str(sample.reference_bundle_path),
            "reference_short_context_path": (
                str(sample.short_context_path) if sample.short_context_path is not None else None
            ),
            "reference_long_context_path": (
                str(sample.long_context_path) if sample.long_context_path is not None else None
            ),
            "candidates": [],
        }
        for sample in reference_manifest.samples
    }

    for candidate_manifest in candidate_manifests:
        if candidate_manifest.configuration in seen_configurations:
            raise ValueError(
                f"duplicate candidate configuration {candidate_manifest.configuration!r} in merged runtime capture."
            )
        seen_configurations.add(candidate_manifest.configuration)
        if candidate_manifest.source_model != reference_manifest.source_model:
            raise ValueError(
                "candidate source_model does not match reference manifest: "
                f"{candidate_manifest.source_model!r} != {reference_manifest.source_model!r}"
            )
        if candidate_manifest.corpus_name != reference_manifest.corpus_name:
            raise ValueError(
                "candidate corpus_name does not match reference manifest: "
                f"{candidate_manifest.corpus_name!r} != {reference_manifest.corpus_name!r}"
            )
        if candidate_manifest.seed != reference_manifest.seed:
            raise ValueError(
                f"candidate seed does not match reference manifest: {candidate_manifest.seed} != {reference_manifest.seed}"
            )

        candidate_by_id = {sample.id: sample for sample in candidate_manifest.samples}
        missing = sorted(set(reference_manifest.sample_ids) - set(candidate_by_id))
        if missing:
            raise ValueError(
                f"candidate manifest {candidate_manifest.manifest_path} is missing sample ids: {', '.join(missing)}."
            )
        extra = sorted(set(candidate_by_id) - set(reference_manifest.sample_ids))
        if extra:
            raise ValueError(
                f"candidate manifest {candidate_manifest.manifest_path} has unknown sample ids: {', '.join(extra)}."
            )

        for reference_sample in reference_manifest.samples:
            candidate_sample = candidate_by_id[reference_sample.id]
            if (
                candidate_sample.prompt is not None
                and candidate_sample.prompt != reference_sample.prompt
            ):
                raise ValueError(
                    f"candidate manifest {candidate_manifest.manifest_path} prompt mismatch for sample "
                    f"{reference_sample.id!r}."
                )
            merged_samples[reference_sample.id]["candidates"].append(
                {
                    "configuration": candidate_manifest.configuration,
                    "bundle_path": str(candidate_sample.bundle_path),
                    "bundle_format": candidate_sample.bundle_format,
                    "captured_components": list(candidate_sample.captured_components),
                    "token_count": candidate_sample.token_count,
                    "short_context_tokens": candidate_sample.short_context_tokens,
                    "candidate_serialized_bytes": candidate_sample.candidate_serialized_bytes,
                    "short_context_path": (
                        str(candidate_sample.short_context_path)
                        if candidate_sample.short_context_path is not None
                        else None
                    ),
                    "long_context_path": (
                        str(candidate_sample.long_context_path)
                        if candidate_sample.long_context_path is not None
                        else None
                    ),
                    "runtime_summary": (
                        dict(candidate_sample.runtime_summary)
                        if candidate_sample.runtime_summary is not None
                        else None
                    ),
                }
            )

    return {
        "run_id": f"{reference_manifest.run_id}-runtime-merge",
        "created_at_utc": _require_utc_timestamp(created_at_utc, context="merged manifest `created_at_utc`"),
        "evidence_tier": max(
            [reference_manifest.evidence_tier, *[candidate.evidence_tier for candidate in candidate_manifests]],
            key=lambda tier: tier.rank,
        ).value,
        "source_model": reference_manifest.source_model,
        "corpus_name": reference_manifest.corpus_name,
        "seed": reference_manifest.seed,
        "commit_sha": reference_manifest.commit_sha,
        "notes": [
            f"reference:{reference_manifest.manifest_path}",
            *[f"candidate:{candidate.configuration}:{candidate.manifest_path}" for candidate in candidate_manifests],
        ],
        "artifact_path": str(artifact_manifest_path) if artifact_manifest_path is not None else None,
        "reference_configuration": reference_manifest.reference_configuration,
        "top_p_threshold": top_p_threshold,
        "reference_provenance": {
            "manifest_path": str(reference_manifest.manifest_path),
            "run_id": reference_manifest.run_id,
            "created_at_utc": reference_manifest.created_at_utc,
            "evidence_tier": reference_manifest.evidence_tier.value,
            "source_model": reference_manifest.source_model,
            "corpus_name": reference_manifest.corpus_name,
            "seed": reference_manifest.seed,
            "commit_sha": reference_manifest.commit_sha,
        },
        "candidate_provenance": [
            candidate.provenance_dict()
            for candidate in candidate_manifests
        ],
        "merged_manifest_path": str(output_manifest_path),
        "samples": list(merged_samples.values()),
    }


__all__ = [
    "ReferenceCaptureManifest",
    "ReferenceCaptureSample",
    "RuntimeCandidateManifest",
    "RuntimeCandidateSample",
    "build_runtime_candidate_manifest_template",
    "merge_runtime_capture_manifests",
]
