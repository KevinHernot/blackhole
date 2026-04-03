# Blackhole scripts

The `scripts/` directory is the Python proof-of-concept surface for Blackhole.

Every script is designed to answer one question:

> **How does this idea improve on the standard `q8_0` architecture?**

These scripts do not claim to be runtime-backed benchmarks. The reported numbers
come from a deterministic scenario model unless a script is explicitly pointed
at a measured artifact.

## Canonical configuration ladder

Every script uses the same eight user-facing configurations and only these eight:

- `f16`
- `q8_0`
- `q8_0 + Semantic PVS`
- `q8_0 + Portal Attention`
- `q8_0 + Predictive Transport`
- `q8_0 + Procedural Weights`
- `q8_0 + Token Merging`
- `blackhole (q8_0 + all 5)`

The common baseline is always `q8_0`.

## How to read the scripts

Each script prints a proof-of-concept table for the ladder above and highlights
the improvement relative to standard `q8_0`.

- `portal_attention.py` shows how portal windows reduce the active KV working set relative to `q8_0`.
- `semantic_pvs_routing.py` shows how Semantic PVS reduces active keys and value reads relative to `q8_0`.
- `predictive_transport.py` shows how predictive deltas reduce cross-layer payload volume relative to `q8_0`.
- `kl_divergence_vs_f16.py` shows how far each configuration moves away from the `f16` reference.
- `measure_skip_rate.py` shows how much skip leverage each configuration gets at long context relative to `q8_0`.
- `long_context_perplexity.py` shows long-context perplexity and stability relative to `q8_0`.
- `compression_quality.py` shows the compression-quality frontier relative to `q8_0`.
- `run_measured_model_eval.py` captures optional reference-side measured-model artifacts for teacher-forced long-context evals.
- `../evals/init_runtime_candidate_manifest.py` prepares candidate capture manifests for the sibling `blackhole_runtime` workspace.
- `niah_pvs_routing.py` shows how many blocks each configuration must inspect to recover the same needles as `q8_0`.
- `niah_test.py` runs the multi-mode NIAH proof of concept over the full ladder.
- `token_merging_poc.py` maps 3D greedy meshing to sequence reduction relative to `q8_0`.
- `unified_poc.py` prints the consolidated view of the full Blackhole stack.

## Quick start

Install the local package:

```bash
python3 -m pip install -e .
```

Run the unified summary:

```bash
python3 scripts/unified_poc.py
```

Run the proof-of-concept NIAH sweep:

```bash
python3 scripts/niah_test.py
```

Refresh the generated docs after a script/test pass:

```bash
python3 scripts/generate_results_docs.py --pytest-status "<latest pytest summary>"
```

The generated [`docs/results.md`](../docs/results.md) and
[`docs/result_summary.md`](../docs/result_summary.md) carry:

- the current UTC manifest
- the latest NIAH artifact paths
- the active quality evidence source (`proxy` vs artifact-backed)

## Notes

- `blackhole_core/comparison_profiles.py` is the source of truth for the ladder and section ordering.
- `blackhole_core/scenario_model.py` is the shared source of truth for the proof-of-concept mechanics.
- `run_measured_model_eval.py` needs the optional `bench` extras (`torch`, `transformers`, `accelerate`) and currently captures the `f16` reference side only.
- Candidate-side measured captures are expected to come from the sibling `blackhole_runtime` workspace through `blackhole_core/runtime_capture_contract.py`.
- `blackhole_runtime` can fill those manifests with observed-output captures (`runtime_observed_v1`) plus short/long context eval files, and the measured-quality builder can then promote honest KL/perplexity artifacts from that handoff.
- `compression_quality.py --source artifact` still requires full tensor-bundle reconstruction/compression metrics; `runtime_observed_v1` artifacts intentionally leave that frontier unavailable until true candidate tensor bundles exist.
- When real runtime captures exist, this repo should prefer ingesting artifacts emitted by the sibling `blackhole_runtime` workspace instead of inventing a parallel runtime path here.
- `scripts/_comparison_profiles.py` and `scripts/_scenario_model.py` remain compatibility shims for existing script entry points.
- All proof-of-concept outputs are ordered the same way so deltas versus `q8_0` stay easy to read.
