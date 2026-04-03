# Blackhole eval inputs

This directory is the home for evaluation inputs and measured-quality artifacts.

Current status:

- `quality_corpus.jsonl` is a small versioned starter corpus for future measured
  quality runs.
- `init_runtime_candidate_manifest.py` prepares a contract-compliant candidate
  capture manifest for `blackhole_runtime`, inheriting the source model, corpus,
  seed, and sample IDs from a reference capture.
- `build_measured_quality_artifact.py` turns a saved bundle/context manifest
  into a measured-quality artifact.
- `merge_runtime_capture_manifests.py` now validates reference and candidate
  capture manifests before it merges them, and it preserves structured
  provenance for the reference capture plus every runtime/backend candidate.
- `generate_measured_quality_fixture.py` now writes a deterministic
  bundle-backed harness fixture:
  - reference and candidate tensor bundles under `evals/bundles/fixture/`
  - per-configuration short/long context eval files under the same bundle dir
  - a measured-quality artifact at `evals/artifacts/measured_quality_fixture.json`
  - a provenance manifest at `evals/artifacts/measured_quality_fixture_manifest.json`
- `scripts/run_measured_model_eval.py` captures reference-side bundles and
  teacher-forced short/long context perplexity inputs from a real Transformers
  causal LM when the optional bench dependencies are installed.

Important:

- The fixture artifact is for harness validation only.
- It is intentionally **not** a headline benchmark artifact and should not be
  used to replace the published PoC numbers in the docs.
- Real promotions from `proxy` to `measured_offline` or `measured_model` should
  use captured tensor bundles or actual model-eval runs rather than these
  synthetic fixture bundles.
- The measured-model runner currently captures the `f16` reference side only;
  candidate-side real-model captures still need to be filled by
  `blackhole_runtime` before the quality docs can be promoted to measured
  results.
- `blackhole_runtime` can now emit observed-output candidate captures with
  real prompt-token logits, token embeddings, and short/long context eval
  files. Those captures are now a supported quality handoff for measured
  KL/perplexity promotion, but they are still not full KV-backed tensor
  bundles, so compression/frontier claims remain unavailable until the runtime
  can export true candidate tensor bundles.
- The preferred long-term source for captured runtime bundles is the sibling
  `blackhole_runtime` project; this repo should validate and summarize those
  artifacts rather than fork the runtime implementation path.
- Candidate manifests consumed from `blackhole_runtime` are now expected to
  include:
  - `source_model`, `corpus_name`, and `seed` that match the reference capture
  - `backend` plus a non-empty `backend_configuration` object
  - the full sample ID set from the reference manifest
  - per-sample bundle/context paths and optional runtime summaries
  - optional `bundle_format`, `captured_components`, `token_count`, and
    `short_context_tokens` when the runtime emitted observed-output captures

Suggested workflow:

```bash
python3 scripts/run_measured_model_eval.py --model <model> ...
python3 evals/init_runtime_candidate_manifest.py \
  --reference-manifest evals/captured/.../captured_reference_manifest.json \
  --configuration "q8_0" \
  --backend ggml \
  --backend-config-json '{"threads": 8, "gpu_layers": 0}' \
  --capture-root evals/captured/runtime_q8_0 \
  --output evals/captured/runtime_q8_0/runtime_q8_0.json

python3 ../blackhole_runtime/benchmarks/fill_runtime_candidate_manifest.py \
  --input-manifest evals/captured/runtime_q8_0/runtime_q8_0.json \
  --output-manifest evals/captured/runtime_q8_0/runtime_q8_0_filled.json \
  --smoke-binary ../blackhole_runtime/build-llama/bin/blackhole-runtime-smoke \
  --model /absolute/path/to/model.gguf \
  --decode-tokens 3 \
  --capture-observed-bundles
```

`blackhole_runtime` can then fill the planned bundle/context paths in that
manifest and emit the completed candidate capture for later merge/build steps.
When those captures use `bundle_format=runtime_observed_v1`, the measured
artifact builder can now promote the honest prompt-logit and context-perplexity
metrics it contains while leaving tensor-bundle-only fields unset.
