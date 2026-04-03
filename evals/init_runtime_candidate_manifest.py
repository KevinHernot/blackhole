#!/usr/bin/env python3
"""Prepare a candidate-side runtime capture manifest for blackhole_runtime."""

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
    EvidenceTier,
    ReferenceCaptureManifest,
    build_runtime_candidate_manifest_template,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-manifest",
        required=True,
        help="Reference manifest emitted by scripts/run_measured_model_eval.py.",
    )
    parser.add_argument(
        "--configuration",
        required=True,
        help="Configuration label for the runtime candidate capture.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        help="Runtime backend name, for example ggml, metal, cuda, or mlx.",
    )
    parser.add_argument(
        "--backend-config-json",
        default=None,
        help="Inline JSON object describing backend configuration knobs.",
    )
    parser.add_argument(
        "--backend-config-file",
        default=None,
        help="Path to a JSON file describing backend configuration knobs.",
    )
    parser.add_argument(
        "--capture-root",
        required=True,
        help="Directory where blackhole_runtime should write bundles and context eval files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the initialized runtime candidate manifest JSON.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run_id for the candidate manifest.",
    )
    parser.add_argument(
        "--commit-sha",
        default=None,
        help="Optional blackhole_runtime commit SHA to stamp into the candidate manifest.",
    )
    parser.add_argument(
        "--evidence-tier",
        choices=(EvidenceTier.MEASURED_MODEL.value, EvidenceTier.RUNTIME_BENCHMARK.value),
        default=EvidenceTier.MEASURED_MODEL.value,
        help="Evidence tier to stamp into the candidate manifest (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _load_backend_configuration(args: argparse.Namespace) -> dict[str, object]:
    if bool(args.backend_config_json) == bool(args.backend_config_file):
        raise SystemExit("Error: provide exactly one of --backend-config-json or --backend-config-file.")
    if args.backend_config_json:
        payload = json.loads(args.backend_config_json)
    else:
        payload = json.loads(Path(args.backend_config_file).expanduser().read_text())
    if not isinstance(payload, dict) or not payload:
        raise SystemExit("Error: backend configuration must be a non-empty JSON object.")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        reference_manifest = ReferenceCaptureManifest.from_path(args.reference_manifest)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    backend_configuration = _load_backend_configuration(args)
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    try:
        manifest = build_runtime_candidate_manifest_template(
            reference_manifest,
            configuration=args.configuration,
            backend=args.backend,
            backend_configuration=backend_configuration,
            capture_root=args.capture_root,
            run_id=args.run_id,
            created_at_utc=created_at,
            evidence_tier=EvidenceTier(args.evidence_tier),
            commit_sha=args.commit_sha,
            notes=(
                "Prepared by blackhole/evals/init_runtime_candidate_manifest.py",
                f"Reference manifest: {reference_manifest.manifest_path}",
            ),
        )
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc

    output_path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")
    print(f"Initialized runtime candidate manifest: {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
