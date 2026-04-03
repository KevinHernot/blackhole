# Blackhole Docs Index

This directory holds the human-readable summaries, the full proof-of-concept results, and the generated artifacts that come out of the script suite.

Each generated document now labels its own evidence tier explicitly, so `proxy`,
`measured_offline`, and future `measured_model` results do not blur together.

## Core documents

- [`result_summary.md`](./result_summary.md) — compact executive summary of the current Blackhole proof-of-concept story.
- [`results.md`](./results.md) — full tables, heatmaps, and cross-pillar analysis.
- [`package_architecture.md`](./package_architecture.md) — algorithm-package architecture, current NumPy PoC contracts, and validation surface.

## Generated docs workflow

Refresh the checked-in docs with:

- `python3 scripts/generate_results_docs.py --pytest-status "<latest pytest summary>"`

That generator stamps a single UTC manifest into [`results.md`](./results.md)
and [`result_summary.md`](./result_summary.md), carries forward the latest
NIAH artifact paths, and records whether quality sections came from proxy or
artifact-backed inputs.

## Generated artifacts

### Threshold ablation

- [`threshold-ablation-logs/skip_rate_proof_of_concept.json`](./threshold-ablation-logs/skip_rate_proof_of_concept.json) — dense decode skip-rate data emitted by `measure_skip_rate.py`.

### NIAH outputs

These artifacts are written to the repo-root `niah_results_poc/` directory.
The current latest JSON/Markdown paths are enumerated inside the generated
[`results.md`](./results.md) and [`result_summary.md`](./result_summary.md)
manifest blocks.

## Where to read the latest run

- [`result_summary.md`](./result_summary.md) carries the current UTC manifest and headline deltas.
- [`results.md`](./results.md) carries the current section outputs and artifact provenance.

## Reading order

1. Start with [`result_summary.md`](./result_summary.md) for the shortest explanation.
2. Use [`results.md`](./results.md) when you want the section-by-section tables and heatmaps.
3. Open the `niah_results_poc/` artifacts when you want the raw markdown/JSON emitted by the latest `niah_test.py` runs.
4. Open [`threshold-ablation-logs/skip_rate_proof_of_concept.json`](./threshold-ablation-logs/skip_rate_proof_of_concept.json) when you want the raw dense skip-rate data.
