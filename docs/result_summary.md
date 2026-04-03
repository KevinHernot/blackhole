# Blackhole PoC — Executive Summary

> **Generated at (UTC):** 2026-04-01T16:34:44Z | **Seed:** 42 | **Baseline:** standard `q8_0`
> **Quality source:** proxy scenario-model outputs
> **Git commit:** `5bd8b89`
> **Verification:** `pytest -q tests` -> **77 passed**

## Headline

Blackhole still reads best as a stack of additive improvements on top of standard `q8_0`: the routing and locality pillars move the speed and retrieval story, while the quality sections now declare their source explicitly as proxy scenario-model outputs.

## Key Metrics — Full Blackhole vs q8_0

| Metric | q8_0 | full Blackhole | Delta |
| --- | --- | --- | --- |
| Prefill speed proxy | 1.00x | 4.21x | 4.21x |
| MoE decode proxy | 1.00x | 2.70x | 2.70x |
| Transport speed proxy | 1.00x | 3.16x | 3.16x |
| NIAH single-needle expected accuracy | 79.7% | 88.4% | +8.7 pts |
| NIAH multi-key expected accuracy | 75.7% | 89.8% | +14.2 pts |
| NIAH multi-value expected accuracy | 71.0% | 88.9% | +17.9 pts |
| KL drift vs f16 | 0.0029 | 0.0006 | -0.0023 |
| Long-context perplexity | 8.01 | 7.64 | -0.37 |
| Compression frontier vs q8_0 | 1.00x | 1.49x | +0.49x |

## Evidence Tiers

- Prefill, decode, transport, and skip-rate sections are still `proxy` scenario-model outputs.
- Quality sections are currently sourced from proxy scenario-model outputs.
- NIAH summary numbers come from the latest generated artifact JSON for each mode.

## Latest NIAH Artifacts

| Mode | JSON | Markdown |
| --- | --- | --- |
| single | niah_results_poc/niah_single_20260401_163422.json | niah_results_poc/niah_single_20260401_163422.md |
| multi-key | niah_results_poc/niah_multi-key_20260401_163422.json | niah_results_poc/niah_multi-key_20260401_163422.md |
| multi-value | niah_results_poc/niah_multi-value_20260401_163422.json | niah_results_poc/niah_multi-value_20260401_163422.md |

## Caveats

- This is still a proof-of-concept suite, not a native Blackhole runtime benchmark.
- Generated docs are now timestamped from one UTC manifest, but headline claims remain bounded by the highest evidence tier available per section.
