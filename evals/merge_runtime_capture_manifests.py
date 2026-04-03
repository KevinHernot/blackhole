#!/usr/bin/env python3
"""Merge reference captures with blackhole_runtime candidate manifests."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core import (  # noqa: E402
    merge_runtime_capture_manifests,
    ReferenceCaptureManifest,
    RuntimeCandidateManifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-manifest",
        required=True,
        help="Reference manifest from scripts/run_measured_model_eval.py.",
    )
    parser.add_argument(
        "--candidate-manifest",
        action="append",
        required=True,
        help="Candidate manifest emitted by blackhole_runtime. Repeat for multiple configurations.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the merged capture manifest JSON.",
    )
    parser.add_argument(
        "--artifact-path",
        default=None,
        help="Optional measured-quality artifact path to embed in the merged manifest.",
    )
    parser.add_argument(
        "--top-p-threshold",
        type=float,
        default=0.90,
        help="Top-p threshold metadata for later artifact building (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        reference_manifest = ReferenceCaptureManifest.from_path(args.reference_manifest)
        candidate_manifests = [
            RuntimeCandidateManifest.from_path(path) for path in args.candidate_manifest
        ]
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        merged_payload = merge_runtime_capture_manifests(
            reference_manifest,
            candidate_manifests,
            created_at_utc=created_at,
            output_path=output_path,
            artifact_path=args.artifact_path,
            top_p_threshold=args.top_p_threshold,
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    output_path.write_text(json.dumps(merged_payload, indent=2) + "\n")
    print(f"Merged runtime capture manifest: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
