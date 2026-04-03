# Blackhole Roadmap

Last updated: 2026-04-01 UTC

This roadmap tracks the Python proof of concept as a q8_0-primary Blackhole
benchmark suite. The repo owns the algorithm, eval, and provenance side; the
sibling `blackhole_runtime` workspace owns executable runtime captures and
backend benchmarking.

## Evidence tiers

Every published result should belong to exactly one tier:

1. `proxy`
2. `measured_offline`
3. `measured_model`
4. `runtime_benchmark`

Headline claims should always use the highest completed tier for a section, and
the docs should label that tier explicitly.

## Phase 0: Clean q8_0 PoC baseline

- [x] Standardize the canonical ladder around `q8_0`
- [x] Remove retired legacy labels from the Python PoC surface
- [x] Make retrieval summaries use expected probabilities instead of single sampled hits
- [x] Add standalone PoC sections for KL, long-context quality, and compression-quality
- [x] Bring docs and tests back into a coherent state

## Phase 1: Measured-quality foundation

- [x] Add explicit evidence-tier metadata types
- [x] Add a measured-quality artifact schema and JSON round-trip support
- [x] Add deterministic fixture generation for the measured-quality harness
- [x] Rebuild the fixture harness from saved bundles and context files instead of direct in-memory metrics
- [x] Add artifact-aware modes to the three quality scripts
- [x] Add a measured-quality docs generator that can consume the artifact schema

Goal:
- Give the repo a real measured artifact contract before promoting any section
  from proxy to measured in the published docs.

## Phase 2: Offline measured quality

- [ ] Add versioned eval inputs under `evals/`
- [ ] Capture or generate reference/candidate tensor bundles for all supported configurations
- [x] Add an observed-output runtime capture handoff for candidate-side logits, token embeddings, and short/long context eval files
- [x] Add a manifest-driven artifact builder for saved bundle/context captures
- [x] Teach `blackhole` to consume `runtime_observed_v1` captures for honest KL/perplexity promotion without over-claiming full KV coverage
- [ ] Convert `KL Divergence vs f16` to `measured_offline`
- [ ] Convert `Compression Quality` to `measured_offline`
- [x] Publish sample counts, artifact manifests, and provenance metadata

Goal:
- Replace proxy KL/cosine/MSE/frontier math with artifact-backed measurements.

## Phase 3: Measured model evals

- [x] Add a reference-side model-eval runner with optional `torch` / `transformers` dependencies
- [ ] Compute real teacher-forced long-context perplexity
- [ ] Add task-level long-context / retrieval evals beyond surrogate metrics
- [ ] Report slice breakdowns and uncertainty intervals

Goal:
- Make `Long-Context Perplexity` and adjacent quality claims come from real model execution.

## Phase 4: Generated reporting

- [x] Generate `docs/results.md` from script/artifact outputs
- [x] Generate `docs/result_summary.md` from the same source-of-truth manifest
- [ ] Add tests that reject stale dates, mismatched artifact paths, or hand-edited measured tables

Goal:
- Eliminate drift between scripts, artifacts, and docs.

## Immediate next steps

1. Capture and publish real `runtime_observed_v1` artifacts for the q8_0/Blackhole ladder so the KL and long-context sections can move off synthetic fixtures.
2. Export true candidate tensor bundles from `blackhole_runtime` so compression-quality promotion can use full reconstruction/compression evidence instead of prompt-observed subsets.
3. Add stricter generated-doc tests that reject stale manifests or measured-table drift.
