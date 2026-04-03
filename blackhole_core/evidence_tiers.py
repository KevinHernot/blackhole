from __future__ import annotations

"""Evidence-tier metadata for Blackhole benchmark outputs."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvidenceTier(str, Enum):
    PROXY = "proxy"
    MEASURED_OFFLINE = "measured_offline"
    MEASURED_MODEL = "measured_model"
    RUNTIME_BENCHMARK = "runtime_benchmark"

    @property
    def rank(self) -> int:
        return {
            EvidenceTier.PROXY: 0,
            EvidenceTier.MEASURED_OFFLINE: 1,
            EvidenceTier.MEASURED_MODEL: 2,
            EvidenceTier.RUNTIME_BENCHMARK: 3,
        }[self]

    def promotes_over(self, other: "EvidenceTier") -> bool:
        return self.rank > other.rank


@dataclass(frozen=True)
class ArtifactMetadata:
    run_id: str
    created_at_utc: str
    evidence_tier: EvidenceTier
    source_model: str
    corpus_name: str
    seed: int
    commit_sha: str | None = None
    notes: tuple[str, ...] = ()

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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArtifactMetadata":
        return cls(
            run_id=str(payload["run_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            evidence_tier=EvidenceTier(str(payload["evidence_tier"])),
            source_model=str(payload["source_model"]),
            corpus_name=str(payload["corpus_name"]),
            seed=int(payload["seed"]),
            commit_sha=str(payload["commit_sha"]) if payload.get("commit_sha") is not None else None,
            notes=tuple(str(note) for note in payload.get("notes", ())),
        )


__all__ = [
    "ArtifactMetadata",
    "EvidenceTier",
]
