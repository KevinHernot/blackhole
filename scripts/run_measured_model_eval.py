#!/usr/bin/env python3
"""Run measured-model reference evals and capture bundle artifacts.

This script is the bridge from the current proof-of-concept suite to a real
teacher-forced long-context evaluation path. It captures:

- reference tensor bundles from a live Transformers causal LM
- aligned short/long context logits + targets for perplexity computation
- a reference-side manifest that future candidate-capture or offline bundle
  builders can extend into measured quality artifacts
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core import (  # noqa: E402
    EvidenceTier,
    TensorBundle,
    perplexity_from_logits,
    save_context_eval,
    save_tensor_bundle,
)

DEFAULT_CORPUS = PROJECT_ROOT / "evals" / "quality_corpus.jsonl"


def _import_bench_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "run_measured_model_eval.py requires optional bench dependencies: torch and transformers."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def _git_commit_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_corpus(path: str | Path, *, max_samples: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "id" not in row or "prompt" not in row:
            raise ValueError("Each corpus row must contain `id` and `prompt`.")
        rows.append({"id": str(row["id"]), "prompt": str(row["prompt"])})
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def _tensor_to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _prepare_inputs_for_device(inputs: dict[str, object], device: str) -> dict[str, object]:
    prepared: dict[str, object] = {}
    for key, value in inputs.items():
        prepared[key] = value.to(device) if hasattr(value, "to") else value
    return prepared


def _bundle_from_outputs(outputs) -> TensorBundle:
    k_tensors = []
    v_tensors = []
    for layer_cache in outputs.past_key_values:
        key, value = tuple(layer_cache)
        k_tensors.append(_tensor_to_numpy(key.squeeze(0)))
        v_tensors.append(_tensor_to_numpy(value.squeeze(0)))

    activations = None
    if getattr(outputs, "hidden_states", None):
        hidden_states = outputs.hidden_states[-3:]
        activations = np.stack([_tensor_to_numpy(state.squeeze(0)) for state in hidden_states])

    logits = None
    if getattr(outputs, "logits", None) is not None:
        logits = _tensor_to_numpy(outputs.logits.squeeze(0))

    return TensorBundle(
        k_cache=np.stack(k_tensors),
        v_cache=np.stack(v_tensors),
        activations=activations,
        logits=logits,
    )


def _aligned_context_eval(logits, input_ids) -> tuple[np.ndarray, np.ndarray]:
    logits_array = _tensor_to_numpy(logits.squeeze(0))
    input_ids_array = _tensor_to_numpy(input_ids.squeeze(0)).astype(int)
    if input_ids_array.shape[0] < 2:
        raise ValueError("Each prompt must tokenize to at least 2 tokens for teacher-forced perplexity.")
    return logits_array[:-1], input_ids_array[1:]


def _forward(model, torch_module, inputs: dict[str, object]):
    no_grad = torch_module.no_grad() if hasattr(torch_module, "no_grad") else nullcontext()
    with no_grad:
        return model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
        )


def _capture_sample(
    *,
    sample: dict[str, str],
    tokenizer,
    model,
    torch_module,
    device: str,
    short_context_tokens: int,
    output_dir: Path,
) -> dict[str, object]:
    encoded = _prepare_inputs_for_device(tokenizer(sample["prompt"], return_tensors="pt"), device)
    full_outputs = _forward(model, torch_module, encoded)

    bundle = _bundle_from_outputs(full_outputs)
    bundle_path = output_dir / "bundles" / f"{sample['id']}__f16_reference.npz"
    save_tensor_bundle(bundle_path, bundle)

    long_logits, long_targets = _aligned_context_eval(full_outputs.logits, encoded["input_ids"])
    long_context_path = output_dir / "contexts" / f"{sample['id']}__long.npz"
    save_context_eval(long_context_path, long_logits, long_targets)
    long_perplexity = perplexity_from_logits(long_logits, long_targets)

    token_count = int(_tensor_to_numpy(encoded["input_ids"]).shape[-1])
    short_length = max(2, min(short_context_tokens, token_count))
    if short_length == token_count:
        short_logits, short_targets = long_logits, long_targets
    else:
        short_inputs = {}
        for key, value in encoded.items():
            short_inputs[key] = value[:, :short_length] if hasattr(value, "__getitem__") else value
        short_outputs = _forward(model, torch_module, short_inputs)
        short_logits, short_targets = _aligned_context_eval(short_outputs.logits, short_inputs["input_ids"])
    short_context_path = output_dir / "contexts" / f"{sample['id']}__short.npz"
    save_context_eval(short_context_path, short_logits, short_targets)
    short_perplexity = perplexity_from_logits(short_logits, short_targets)

    return {
        "id": sample["id"],
        "prompt": sample["prompt"],
        "token_count": token_count,
        "short_context_tokens": short_length,
        "reference_bundle_path": str(bundle_path),
        "short_context_path": str(short_context_path),
        "long_context_path": str(long_context_path),
        "short_context_perplexity": short_perplexity,
        "long_context_perplexity": long_perplexity,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Transformers model name or local path.")
    parser.add_argument(
        "--corpus",
        default=str(DEFAULT_CORPUS),
        help="JSONL corpus with `id` and `prompt` fields (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "evals" / "captured" / "measured_model_eval"),
        help="Directory for bundles, context evals, summary JSON, and manifest output.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on (default: %(default)s).",
    )
    parser.add_argument(
        "--short-context-tokens",
        type=int,
        default=128,
        help="Token budget for the short-context perplexity slice (default: %(default)s).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of corpus rows to run.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to the tokenizer/model loaders.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    (output_dir / "bundles").mkdir(parents=True, exist_ok=True)
    (output_dir / "contexts").mkdir(parents=True, exist_ok=True)

    corpus = _load_corpus(args.corpus, max_samples=args.max_samples)
    if not corpus:
        raise SystemExit("Error: corpus is empty.")

    try:
        torch_module, auto_model, auto_tokenizer = _import_bench_dependencies()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    tokenizer = auto_tokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = auto_model.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if hasattr(model, "to"):
        model = model.to(args.device)
    if hasattr(model, "eval"):
        model.eval()

    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sample_results = [
        _capture_sample(
            sample=sample,
            tokenizer=tokenizer,
            model=model,
            torch_module=torch_module,
            device=args.device,
            short_context_tokens=args.short_context_tokens,
            output_dir=output_dir,
        )
        for sample in corpus
    ]

    avg_short_ppl = sum(sample["short_context_perplexity"] for sample in sample_results) / len(sample_results)
    avg_long_ppl = sum(sample["long_context_perplexity"] for sample in sample_results) / len(sample_results)

    summary_path = output_dir / "measured_model_eval.json"
    summary_payload = {
        "run_id": f"measured-model-{created_at.replace(':', '').replace('-', '')}",
        "created_at_utc": created_at,
        "evidence_tier": EvidenceTier.MEASURED_MODEL.value,
        "source_model": args.model,
        "corpus_name": Path(args.corpus).name,
        "device": args.device,
        "short_context_tokens": args.short_context_tokens,
        "commit_sha": _git_commit_sha(),
        "sample_count": len(sample_results),
        "average_short_context_perplexity": avg_short_ppl,
        "average_long_context_perplexity": avg_long_ppl,
        "samples": sample_results,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

    manifest_path = output_dir / "captured_reference_manifest.json"
    manifest_payload = {
        "run_id": summary_payload["run_id"],
        "created_at_utc": created_at,
        "evidence_tier": EvidenceTier.MEASURED_MODEL.value,
        "source_model": args.model,
        "corpus_name": Path(args.corpus).name,
        "seed": 0,
        "commit_sha": summary_payload["commit_sha"],
        "notes": [
            "Reference-side measured model capture.",
            "Candidate bundle captures still need to be added before building a full measured-quality artifact.",
        ],
        "reference_configuration": "f16",
        "samples": sample_results,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")

    print(f"Saved measured-model summary to: {summary_path}")
    print(f"Saved reference capture manifest to: {manifest_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
