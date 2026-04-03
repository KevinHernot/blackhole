from __future__ import annotations

"""Manifest helpers for generated Blackhole result docs."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class NIAHArtifact:
    mode: str
    timestamp: str
    json_path: Path
    markdown_path: Path


@dataclass(frozen=True)
class RunManifest:
    generated_at_utc: str
    baseline: str
    seed: int
    pytest_status: str | None
    quality_source: str
    quality_artifact: Path | None
    commit_sha: str | None
    niah_artifacts: tuple[NIAHArtifact, ...]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def discover_latest_niah_artifacts(output_dir: str | Path) -> tuple[NIAHArtifact, ...]:
    root = Path(output_dir)
    latest: dict[str, tuple[str, Path]] = {}

    for path in root.glob("niah_*.json"):
        stem = path.stem
        if not stem.startswith("niah_"):
            continue
        parts = stem[len("niah_") :].split("_")
        if len(parts) < 3:
            continue
        mode = "_".join(parts[:-2])
        timestamp = f"{parts[-2]}_{parts[-1]}"
        current = latest.get(mode)
        if current is None or timestamp > current[0]:
            latest[mode] = (timestamp, path)

    artifacts: list[NIAHArtifact] = []
    for mode in ("single", "multi-key", "multi-value"):
        current = latest.get(mode)
        if current is None:
            continue
        timestamp, json_path = current
        markdown_path = json_path.with_suffix(".md")
        if markdown_path.exists():
            artifacts.append(
                NIAHArtifact(
                    mode=mode,
                    timestamp=timestamp,
                    json_path=json_path,
                    markdown_path=markdown_path,
                )
            )
    return tuple(artifacts)


__all__ = [
    "NIAHArtifact",
    "RunManifest",
    "discover_latest_niah_artifacts",
    "utc_now_iso",
]
