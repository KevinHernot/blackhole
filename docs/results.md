# Blackhole Proof-of-Concept Results

> **Generated at (UTC):** 2026-04-01T16:34:44Z
> **Baseline:** standard `q8_0`
> **Seed:** 42
> **Quality source:** proxy scenario-model outputs
> **Git commit:** `5bd8b89`
> **Verification:** `pytest -q tests` -> **77 passed**

## Provenance

Evidence tiers in this repo are intentionally split into `proxy`, `measured_offline`, `measured_model`, and `runtime_benchmark`. These generated docs report the highest tier actually provided for each section instead of mixing copied numbers from different runs.

## Latest NIAH Artifacts

- `single`: `niah_results_poc/niah_single_20260401_163422.json` and `niah_results_poc/niah_single_20260401_163422.md`
- `multi-key`: `niah_results_poc/niah_multi-key_20260401_163422.json` and `niah_results_poc/niah_multi-key_20260401_163422.md`
- `multi-value`: `niah_results_poc/niah_multi-value_20260401_163422.json` and `niah_results_poc/niah_multi-value_20260401_163422.md`

## 1. Unified Top-of-Tree Summary

This is the stitched-together Blackhole story: every row uses the same q8_0-primary ladder, and every delta is explained relative to the same standard q8_0 baseline.

### Top-of-Tree Results
Full proof-of-concept ladder comparing every Blackhole configuration against standard q8_0.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: compression, speed proxy, quality proxy, note
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | Compression | Prefill proxy | MoE decode proxy | NIAH proxy | Transport proxy | Quality proxy | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f16 | 1.0x | 0.95x | 0.84x | 0.93x | 0.90x | 1.000 | Full-precision dense reference. |
| q8_0 | 1.9x | 1.00x | 1.00x | 1.00x | 1.00x | 0.994 | Quantized baseline before Blackhole-specific ideas. |
| q8_0 + Semantic PVS | 1.9x | 1.34x | 1.72x | 1.42x | 1.08x | 0.996 | Adds macro-routing on top of q8_0. |
| q8_0 + Portal Attention | 1.9x | 2.57x | 1.47x | 1.14x | 1.15x | 0.996 | Adds domain-local portal windows on top of q8_0. |
| q8_0 + Predictive Transport | 1.9x | 1.09x | 1.09x | 1.08x | 1.24x | 0.995 | Adds lighter layer-to-layer transport on top of q8_0. |
| q8_0 + Procedural Weights | 2.3x | 1.08x | 1.07x | 1.06x | 1.18x | 0.998 | Improves the compression-quality frontier on top of q8_0. |
| q8_0 + Token Merging | 2.2x | 1.28x | 1.18x | 1.00x | 1.62x | 0.992 | Adds sequence-dimension 'Greedy Meshing' on top of q8_0. |
| blackhole (q8_0 + all 5) | 2.8x | 4.21x | 2.70x | 1.83x | 3.16x | 0.999 | Stacks all five Blackhole pillars on top of q8_0. |

Section coverage catalog
| Result section | Configuration ladder |
| --- | --- |
| Top-of-Tree Results | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Prefill Context Scaling (Verified 2K-32K) | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Decode Speed — MoE | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| NIAH Retrieval | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| KL Divergence vs f16 | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Decode Speed — Dense | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Long-Context Perplexity (Primary Quality Metric) | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Compression Quality (Python Prototype) | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Speed Optimization Journey | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |
| Sequence Compression — Greedy Meshing | f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5) |

Takeaway: q8_0 is the common compressed baseline. Blackhole is not a different baseline; it is the set of ideas that make q8_0 better, one pillar at a time and then all five together.

## 2. Portal Attention — Prefill Context Scaling

Scenario: a 3-domain prompt where standard q8_0 keeps the full routed window live, while Portal Attention shrinks the active KV working set to sinks + bridge + active domain.

### Prefill Context Scaling (Verified 2K-32K)
Portal and routing ideas are presented as direct improvements on top of the standard q8_0 prefill story.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: active KV, prefill proxy, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | Active KV tokens | KV reduction | Prefill speed proxy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- |
| f16 | 6144 | 0.0% | 0.95x | 0.95x | Dense fp16 reference — no routing or portal shrinkage. |
| q8_0 | 6144 | 0.0% | 1.00x | 1.00x | Quantized baseline with no routing or portal shrinkage yet. |
| q8_0 + Semantic PVS | 4096 | 33.3% | 1.34x | 1.34x | Semantic routing prunes cold regions before the portal window is built. |
| q8_0 + Portal Attention | 2240 | 63.5% | 2.57x | 2.57x | Portal frustum keeps only sinks, bridge tokens, and the active domain. |
| q8_0 + Predictive Transport | 6144 | 0.0% | 1.09x | 1.09x | Transport gets lighter, but the active prefill window is still q8_0-sized. |
| q8_0 + Procedural Weights | 6144 | 0.0% | 1.08x | 1.08x | Better layout helps the same q8_0 window decode a bit more cheaply. |
| q8_0 + Token Merging | 3968 | 35.4% | 1.28x | 1.28x | Token Merging reduces the absolute number of tokens to scan before portals even apply. |
| blackhole (q8_0 + all 5) | 1216 | 80.2% | 4.21x | 4.21x | All five pillars stack: routing plus portals plus transport plus procedural layouts plus token merging. |

Takeaway: standard q8_0 is already a strong compressed baseline, but Blackhole starts to win once it shrinks *which* tokens stay active. Portal Attention is the first sharp jump; the full Blackhole stack compounds that win instead of stopping there.

## 3. Semantic PVS Routing — MoE Decode Speed

Scenario: a 16K MoE decode step where standard q8_0 still examines the full routed token set, while Semantic PVS culls entire semantic blocks before value reads begin.

### Decode Speed — MoE
Every configuration is shown on the same MoE decode ladder so the gains over q8_0 are explicit.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: active keys, values processed, decode proxy, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | Active key tokens | Values processed | Compute reduction | Decode proxy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- | --- |
| f16 | 16384 | 13345 | 18.55% | 0.84x | 0.84x | Dense fp16 reference. |
| q8_0 | 16384 | 12288 | 25.00% | 1.00x | 1.00x | Quantized baseline with no Blackhole routing yet. |
| q8_0 + Semantic PVS | 4096 | 2785 | 83.00% | 1.72x | 1.72x | Macro-routing keeps only the most relevant semantic blocks. |
| q8_0 + Portal Attention | 6144 | 4301 | 73.75% | 1.47x | 1.47x | Portal windows reduce domain spill but do less macro-culling than Semantic PVS. |
| q8_0 + Predictive Transport | 16384 | 11796 | 28.00% | 1.09x | 1.09x | The same routed work becomes cheaper to move across layers. |
| q8_0 + Procedural Weights | 16384 | 11633 | 29.00% | 1.07x | 1.07x | Procedural reconstruction improves the work done per surviving value read. |
| q8_0 + Token Merging | 10752 | 8064 | 50.78% | 1.18x | 1.18x | Merged tokens reduce the total key-set dimension before routing filters begin. |
| blackhole (q8_0 + all 5) | 2048 | 1024 | 93.75% | 2.70x | 2.70x | Routing, portals, transport, token merging, and procedural weights compound instead of competing. |

Takeaway: standard q8_0 is a strong cache baseline, but Semantic PVS changes *which blocks survive at all*. That makes it the clearest routing upgrade on top of q8_0, and the full Blackhole stack pushes the same idea even further.

## 4. Predictive Transport — Layer Handoff Efficiency

Scenario: a 128-layer pipeline where standard q8_0 still copies full per-layer payloads, while Predictive Transport ships deltas and lets downstream layers reconstruct the next state.

### Speed Optimization Journey
Incremental view of how each Blackhole pillar improves the standard q8_0 architecture.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: speed proxy, vs q8_0, incremental note
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | Transported volume | Reduction vs f16 | Speed proxy vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- |
| f16 | 2.00 GB | 0.0% | 0.90x | Native reference — no transport approximation. |
| q8_0 | 1.80 GB | 10.0% | 1.00x | Quantized baseline, still sending dense layer payloads. |
| q8_0 + Semantic PVS | 1.66 GB | 17.0% | 1.08x | Fewer routed blocks means less payload before transport even starts. |
| q8_0 + Portal Attention | 1.57 GB | 21.5% | 1.15x | Smaller active windows cut the amount of state that must cross layers. |
| q8_0 + Predictive Transport | 1.45 GB | 27.5% | 1.24x | Predicted deltas replace full payload copies while keeping cosine ≈ 0.999. |
| q8_0 + Procedural Weights | 1.52 GB | 24.0% | 1.18x | Structured residuals compress more cleanly than plain q8_0 deltas. |
| q8_0 + Token Merging | 1.11 GB | 44.5% | 1.62x | Merging tokens shrinks the absolute transport payload volume before prediction deltas are even computed. |
| blackhole (q8_0 + all 5) | 0.57 GB | 71.5% | 3.16x | All five pillars align: less state, fewer tokens, better deltas, better reconstruction. |

Takeaway: standard q8_0 already moves less state than fp16, but Predictive Transport is where Blackhole starts shrinking the *handoff itself*. The full stack wins because routing and portals reduce what must move, and predictive deltas reduce how much each surviving step costs.

## 5. Sparse-V Skip-Rate — Dense Decode

Scenario: dense decode at increasing context lengths, with standard q8_0 as the gating baseline and Blackhole variants showing how much more inactive value mass can be skipped safely.

### Decode Speed — Dense
Dense-model proof of concept showing how each Blackhole pillar changes the q8_0 decode budget.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: skip leverage, decode proxy, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | 512 | 2K | 4K | 8K | 16K | 32K | Average | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f16 | 7.6% | 22.1% | 29.3% | 36.6% | 43.8% | 47.6% | 31.2% | 0.85x | Dense fp16 reference. |
| q8_0 | 9.0% | 26.0% | 34.5% | 43.0% | 51.5% | 56.0% | 36.7% | 1.00x | Quantized baseline, still using the same dense gating surface throughout decode. |
| q8_0 + Semantic PVS | 16.0% | 33.0% | 41.5% | 50.0% | 58.5% | 63.0% | 43.7% | 1.19x | Routing sharpens relevance, so more cold values can be skipped safely. |
| q8_0 + Portal Attention | 14.0% | 31.0% | 39.5% | 48.0% | 56.5% | 61.0% | 41.7% | 1.14x | Portal windows keep the active domain tight, increasing skip leverage. |
| q8_0 + Predictive Transport | 12.0% | 29.0% | 37.5% | 46.0% | 54.5% | 59.0% | 39.7% | 1.08x | Transport gains do not change sparsity directly, but they make surviving reads cheaper. |
| q8_0 + Procedural Weights | 13.0% | 30.0% | 38.5% | 47.0% | 55.5% | 60.0% | 40.7% | 1.11x | Procedural structure makes the same dense skip decisions more stable. |
| q8_0 + Token Merging | 9.0% | 26.0% | 34.5% | 43.0% | 51.5% | 56.0% | 36.7% | 1.00x | Merging tokens increases the per-token salience, making skip thresholds more effective. |
| blackhole (q8_0 + all 5) | 34.0% | 51.0% | 59.5% | 68.0% | 76.5% | 81.0% | 61.7% | 1.68x | All five pillars combine into the strongest skip leverage on top of q8_0. |

| Configuration | Average skip leverage | Decode proxy vs q8_0 | Interpretation |
| --- | --- | --- | --- |
| f16 | 31.2% | 0.85x | Dense fp16 reference. |
| q8_0 | 36.7% | 1.00x | Quantized baseline, still using the same dense gating surface throughout decode. |
| q8_0 + Semantic PVS | 43.7% | 1.19x | Routing sharpens relevance, so more cold values can be skipped safely. |
| q8_0 + Portal Attention | 41.7% | 1.14x | Portal windows keep the active domain tight, increasing skip leverage. |
| q8_0 + Predictive Transport | 39.7% | 1.08x | Transport gains do not change sparsity directly, but they make surviving reads cheaper. |
| q8_0 + Procedural Weights | 40.7% | 1.11x | Procedural structure makes the same dense skip decisions more stable. |
| q8_0 + Token Merging | 36.7% | 1.00x | Merging tokens increases the per-token salience, making skip thresholds more effective. |
| blackhole (q8_0 + all 5) | 61.7% | 1.68x | All five pillars combine into the strongest skip leverage on top of q8_0. |

## 6. Token Merging (Greedy Meshing) — Sequence Compression

Scenario: Adapting 3D rendering 'Greedy Meshing' to LLMs. Instead of just shrinking the bit-depth of the cache, we geometrically merge redundant adjacent tokens to shrink the sequence length.

### Sequence Compression — Greedy Meshing
Token Merging proof of concept mapping 3D greedy meshing to sequence reduction.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: original length, merged length, sequence reduction, speed proxy, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

| Configuration | Original length | Merged length | Sequence reduction | Prefill proxy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- | --- |
| f16 | 16384 | 16384 | 0% | 0.95x | 0.95x | Uncompressed dense reference. |
| q8_0 | 16384 | 16384 | 0% | 1.00x | 1.00x | 8-bit baseline before sequence-dimension ideas. |
| q8_0 + Semantic PVS | 16384 | 16384 | 0% | 1.34x | 1.34x | Prunes blocks, but does not fuse tokens within blocks. |
| q8_0 + Portal Attention | 16384 | 16384 | 0% | 2.57x | 2.57x | Shrinks the active window, but keeps it dense. |
| q8_0 + Predictive Transport | 16384 | 16384 | 0% | 1.09x | 1.09x | Lighter layer hand-offs, sequence remains standard q8_0. |
| q8_0 + Procedural Weights | 16384 | 16384 | 0% | 1.08x | 1.08x | Procedural reconstruction helps quality, not sequence length. |
| q8_0 + Token Merging | 16384 | 10649 | 35% | 1.28x | 1.28x | Fuses redundant tokens, shrinking the sequence by ~35%. |
| blackhole (q8_0 + all 5) | 16384 | 10649 | 35% | 4.21x | 4.21x | All 5 pillars stack: bits compressed, sequence merged, irrelevant blocks routed. |

Takeaway: Token Merging is the sequence-dimension equivalent of 3D 'Greedy Meshing'. By merging similar adjacent tokens, we reduce the payload for attention and transport, stacking multiplicatively with the rest of the Blackhole working-state reductions.

## 7. NIAH Retrieval — single

Source artifact: `niah_results_poc/niah_single_20260401_163422.md`

Mode: single | Seed: 42 | 2026-04-01 16:34 UTC

### NIAH Retrieval
Retrieval proofs of concept compare the full Blackhole ladder instead of hiding behind runtime subsets.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: blocks scanned, needles routed, vs q8_0, note
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

Any hit/miss examples below are illustrative deterministic draws from the scenario model.
All ranking tables and average-accuracy summaries use expected hit rates from `retrieval_probability()` so the retrieval story is not driven by single-sample noise.

## Single Needle Retrieval: f16

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ✅ | ❌ |
|  10% | ✅ | ✅ | ❌ | ✅ |
|  20% | ✅ | ❌ | ✅ | ✅ |
|  30% | ✅ | ✅ | ❌ | ❌ |
|  40% | ✅ | ✅ | ✅ | ❌ |
|  50% | ✅ | ✅ | ❌ | ❌ |
|  60% | ✅ | ✅ | ✅ | ✅ |
|  70% | ✅ | ✅ | ✅ | ✅ |
|  80% | ✅ | ✅ | ❌ | ✅ |
|  90% | ✅ | ❌ | ✅ | ✅ |
| 100% | ✅ | ✅ | ❌ | ✅ |

**Illustrative sample: 33/44 hits (75.0%) | Expected mean hit rate: 77.7%**

## Single Needle Retrieval: q8_0

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ✅ | ❌ |
|  10% | ✅ | ❌ | ✅ | ✅ |
|  20% | ✅ | ✅ | ✅ | ✅ |
|  30% | ✅ | ✅ | ✅ | ✅ |
|  40% | ✅ | ❌ | ✅ | ❌ |
|  50% | ✅ | ✅ | ✅ | ❌ |
|  60% | ❌ | ❌ | ✅ | ✅ |
|  70% | ✅ | ✅ | ✅ | ❌ |
|  80% | ✅ | ✅ | ❌ | ✅ |
|  90% | ✅ | ✅ | ❌ | ✅ |
| 100% | ✅ | ✅ | ✅ | ✅ |

**Illustrative sample: 34/44 hits (77.3%) | Expected mean hit rate: 79.7%**

## Single Needle Retrieval: q8_0 + Semantic PVS

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ❌ | ✅ |
|  10% | ✅ | ✅ | ✅ | ❌ |
|  20% | ✅ | ✅ | ✅ | ✅ |
|  30% | ✅ | ✅ | ✅ | ✅ |
|  40% | ✅ | ✅ | ✅ | ✅ |
|  50% | ✅ | ✅ | ✅ | ✅ |
|  60% | ✅ | ❌ | ✅ | ✅ |
|  70% | ✅ | ✅ | ✅ | ✅ |
|  80% | ✅ | ✅ | ❌ | ✅ |
|  90% | ✅ | ✅ | ❌ | ❌ |
| 100% | ✅ | ✅ | ✅ | ❌ |

**Illustrative sample: 37/44 hits (84.1%) | Expected mean hit rate: 86.6%**

## Single Needle Retrieval: q8_0 + Portal Attention

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ✅ | ✅ |
|  10% | ✅ | ❌ | ✅ | ❌ |
|  20% | ✅ | ✅ | ✅ | ✅ |
|  30% | ✅ | ✅ | ❌ | ✅ |
|  40% | ✅ | ✅ | ✅ | ✅ |
|  50% | ❌ | ✅ | ✅ | ✅ |
|  60% | ✅ | ✅ | ❌ | ✅ |
|  70% | ✅ | ✅ | ✅ | ✅ |
|  80% | ✅ | ✅ | ✅ | ❌ |
|  90% | ✅ | ✅ | ✅ | ✅ |
| 100% | ✅ | ✅ | ❌ | ✅ |

**Illustrative sample: 37/44 hits (84.1%) | Expected mean hit rate: 84.4%**

## Single Needle Retrieval: q8_0 + Predictive Transport

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ❌ | ✅ | ✅ | ❌ |
|  10% | ✅ | ✅ | ✅ | ❌ |
|  20% | ❌ | ✅ | ✅ | ❌ |
|  30% | ❌ | ❌ | ✅ | ✅ |
|  40% | ❌ | ✅ | ✅ | ✅ |
|  50% | ✅ | ✅ | ✅ | ✅ |
|  60% | ✅ | ✅ | ✅ | ✅ |
|  70% | ❌ | ✅ | ❌ | ✅ |
|  80% | ✅ | ✅ | ✅ | ❌ |
|  90% | ✅ | ✅ | ✅ | ✅ |
| 100% | ❌ | ✅ | ❌ | ❌ |

**Illustrative sample: 30/44 hits (68.2%) | Expected mean hit rate: 82.7%**

## Single Needle Retrieval: q8_0 + Procedural Weights

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ❌ | ✅ | ✅ |
|  10% | ✅ | ✅ | ✅ | ✅ |
|  20% | ✅ | ✅ | ✅ | ✅ |
|  30% | ❌ | ✅ | ✅ | ✅ |
|  40% | ✅ | ✅ | ✅ | ❌ |
|  50% | ✅ | ✅ | ✅ | ✅ |
|  60% | ✅ | ✅ | ✅ | ✅ |
|  70% | ✅ | ✅ | ✅ | ✅ |
|  80% | ✅ | ✅ | ✅ | ✅ |
|  90% | ✅ | ✅ | ❌ | ✅ |
| 100% | ✅ | ❌ | ✅ | ❌ |

**Illustrative sample: 38/44 hits (86.4%) | Expected mean hit rate: 83.2%**

## Single Needle Retrieval: q8_0 + Token Merging

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ✅ | ✅ |
|  10% | ✅ | ✅ | ❌ | ✅ |
|  20% | ❌ | ✅ | ✅ | ❌ |
|  30% | ✅ | ✅ | ✅ | ❌ |
|  40% | ✅ | ✅ | ❌ | ✅ |
|  50% | ✅ | ✅ | ✅ | ✅ |
|  60% | ✅ | ✅ | ❌ | ❌ |
|  70% | ✅ | ✅ | ✅ | ❌ |
|  80% | ✅ | ✅ | ✅ | ✅ |
|  90% | ❌ | ✅ | ❌ | ❌ |
| 100% | ✅ | ✅ | ✅ | ❌ |

**Illustrative sample: 32/44 hits (72.7%) | Expected mean hit rate: 77.2%**

## Single Needle Retrieval: blackhole (q8_0 + all 5)

| Depth | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
|   0% | ✅ | ✅ | ✅ | ✅ |
|  10% | ✅ | ✅ | ❌ | ✅ |
|  20% | ✅ | ✅ | ✅ | ❌ |
|  30% | ✅ | ✅ | ✅ | ✅ |
|  40% | ✅ | ✅ | ✅ | ✅ |
|  50% | ✅ | ✅ | ✅ | ✅ |
|  60% | ✅ | ❌ | ✅ | ✅ |
|  70% | ✅ | ✅ | ✅ | ✅ |
|  80% | ✅ | ✅ | ✅ | ✅ |
|  90% | ❌ | ✅ | ✅ | ✅ |
| 100% | ✅ | ✅ | ✅ | ✅ |

**Illustrative sample: 40/44 hits (90.9%) | Expected mean hit rate: 88.4%**

## Expected retrieval summary

| Configuration | Expected average accuracy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- |
| f16 | 77.7% | -2.0 pts | Full-precision dense reference. |
| q8_0 | 79.7% | baseline | Quantized baseline before Blackhole-specific improvements. |
| q8_0 + Semantic PVS | 86.6% | +6.9 pts | Semantic routing narrows the retrieval frontier before answer synthesis. |
| q8_0 + Portal Attention | 84.4% | +4.7 pts | Portal locality keeps the active domain tight around the answer region. |
| q8_0 + Predictive Transport | 82.7% | +3.0 pts | Transport stabilization helps the right evidence survive the decode path. |
| q8_0 + Procedural Weights | 83.2% | +3.5 pts | Procedural layouts preserve compressed recall more faithfully than plain q8_0. |
| q8_0 + Token Merging | 77.2% | -2.5 pts | Sequence-dimension merging increases per-token salience before synthesis. |
| blackhole (q8_0 + all 5) | 88.4% | +8.7 pts | All five Blackhole pillars compound on top of q8_0. |

## 8. NIAH Retrieval — multi-key

Source artifact: `niah_results_poc/niah_multi-key_20260401_163422.md`

Mode: multi-key | Seed: 42 | 2026-04-01 16:34 UTC

### NIAH Retrieval
Retrieval proofs of concept compare the full Blackhole ladder instead of hiding behind runtime subsets.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: blocks scanned, needles routed, vs q8_0, note
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

Any hit/miss examples below are illustrative deterministic draws from the scenario model.
All ranking tables and average-accuracy summaries use expected hit rates from `retrieval_probability()` so the retrieval story is not driven by single-sample noise.

## Multi-Key Retrieval (MK-NIAH)

| Configuration | 4K   | 8K   | 16K  | 32K  |
|---|---|---|---|---|
| f16 |  80.9% |  76.0% |  71.2% |  66.4% |
| q8_0 |  82.4% |  77.9% |  73.4% |  68.9% |
| q8_0 + Semantic PVS |  88.4% |  85.2% |  82.1% |  78.9% |
| q8_0 + Portal Attention |  86.1% |  82.6% |  79.1% |  75.6% |
| q8_0 + Predictive Transport |  85.4% |  81.2% |  77.1% |  72.9% |
| q8_0 + Procedural Weights |  85.6% |  81.7% |  77.9% |  74.0% |
| q8_0 + Token Merging |  83.4% |  78.9% |  74.4% |  69.9% |
| blackhole (q8_0 + all 5) |  92.9% |  90.8% |  88.8% |  86.8% |

## Expected retrieval summary

| Configuration | Expected average accuracy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- |
| f16 | 73.6% | -2.0 pts | Full-precision dense reference. |
| q8_0 | 75.7% | baseline | Quantized baseline before Blackhole-specific improvements. |
| q8_0 + Semantic PVS | 83.7% | +8.0 pts | Semantic routing narrows the retrieval frontier before answer synthesis. |
| q8_0 + Portal Attention | 80.8% | +5.2 pts | Portal locality keeps the active domain tight around the answer region. |
| q8_0 + Predictive Transport | 79.1% | +3.5 pts | Transport stabilization helps the right evidence survive the decode path. |
| q8_0 + Procedural Weights | 79.8% | +4.2 pts | Procedural layouts preserve compressed recall more faithfully than plain q8_0. |
| q8_0 + Token Merging | 76.6% | +1.0 pts | Sequence-dimension merging increases per-token salience before synthesis. |
| blackhole (q8_0 + all 5) | 89.8% | +14.2 pts | All five Blackhole pillars compound on top of q8_0. |

## 9. NIAH Retrieval — multi-value

Source artifact: `niah_results_poc/niah_multi-value_20260401_163422.md`

Mode: multi-value | Seed: 42 | 2026-04-01 16:34 UTC

### NIAH Retrieval
Retrieval proofs of concept compare the full Blackhole ladder instead of hiding behind runtime subsets.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: blocks scanned, needles routed, vs q8_0, note
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)

Any hit/miss examples below are illustrative deterministic draws from the scenario model.
All ranking tables and average-accuracy summaries use expected hit rates from `retrieval_probability()` so the retrieval story is not driven by single-sample noise.

## Multi-Value Retrieval (MV-NIAH)

### f16

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  80.8% |  76.0% |  71.2% |  66.3% |
|      4 |  77.0% |  72.1% |  67.3% |  62.5% |
|      8 |  70.4% |  65.6% |  60.7% |  55.9% |

### q8_0

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  82.5% |  78.0% |  73.5% |  69.0% |
|      4 |  78.7% |  74.2% |  69.7% |  65.2% |
|      8 |  72.2% |  67.7% |  63.2% |  58.7% |

### q8_0 + Semantic PVS

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  88.4% |  85.2% |  82.1% |  78.9% |
|      4 |  85.4% |  82.2% |  79.0% |  75.8% |
|      8 |  80.0% |  76.8% |  73.6% |  70.5% |

### q8_0 + Portal Attention

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  86.0% |  82.5% |  79.0% |  75.4% |
|      4 |  82.3% |  78.8% |  75.3% |  71.7% |
|      8 |  75.8% |  72.3% |  68.8% |  65.3% |

### q8_0 + Predictive Transport

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  85.6% |  81.5% |  77.3% |  73.1% |
|      4 |  82.2% |  78.0% |  73.8% |  69.7% |
|      8 |  76.2% |  72.1% |  67.9% |  63.7% |

### q8_0 + Procedural Weights

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  85.9% |  82.0% |  78.2% |  74.3% |
|      4 |  82.6% |  78.8% |  74.9% |  71.1% |
|      8 |  77.1% |  73.3% |  69.4% |  65.6% |

### q8_0 + Token Merging

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  85.5% |  81.0% |  76.5% |  72.0% |
|      4 |  81.7% |  77.2% |  72.7% |  68.2% |
|      8 |  75.2% |  70.7% |  66.2% |  61.7% |

### blackhole (q8_0 + all 5)

| Values | 4K     | 8K     | 16K    | 32K    |
|---|---|---|---|---|
|      2 |  95.0% |  93.0% |  91.0% |  89.0% |
|      4 |  92.6% |  90.6% |  88.5% |  86.5% |
|      8 |  88.3% |  86.3% |  84.2% |  82.2% |


## Expected retrieval summary

| Configuration | Expected average accuracy | vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- |
| f16 | 68.8% | -2.2 pts | Full-precision dense reference. |
| q8_0 | 71.0% | baseline | Quantized baseline before Blackhole-specific improvements. |
| q8_0 + Semantic PVS | 79.8% | +8.8 pts | Semantic routing narrows the retrieval frontier before answer synthesis. |
| q8_0 + Portal Attention | 76.1% | +5.1 pts | Portal locality keeps the active domain tight around the answer region. |
| q8_0 + Predictive Transport | 75.1% | +4.1 pts | Transport stabilization helps the right evidence survive the decode path. |
| q8_0 + Procedural Weights | 76.1% | +5.1 pts | Procedural layouts preserve compressed recall more faithfully than plain q8_0. |
| q8_0 + Token Merging | 74.0% | +3.0 pts | Sequence-dimension merging increases per-token salience before synthesis. |
| blackhole (q8_0 + all 5) | 88.9% | +17.9 pts | All five Blackhole pillars compound on top of q8_0. |

## 10. KL Divergence vs f16

Scenario: compare each configuration's modeled output drift against the full-precision f16 reference. Lower KLD is better; same-top-p estimates how often the candidate keeps the fp16 top-probability set intact.

### KL Divergence vs f16
Quality-proxy section measuring how far each variant drifts from the full-precision reference.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: mean KLD, same-top-p, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Source: deterministic scenario-model proxy.

| Configuration | Mean KLD vs f16 | Same-top-p | Relative drift vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- |
| f16 | 0.0000 | 100.0% | 0.00x | Full-precision dense reference — zero modeled drift by definition. |
| q8_0 | 0.0029 | 99.4% | 1.00x | Quantization alone preserves most of the fp16 decision surface. |
| q8_0 + Semantic PVS | 0.0020 | 99.6% | 0.69x | Routing trims irrelevant work while staying closer to the fp16 surface than plain q8_0. |
| q8_0 + Portal Attention | 0.0020 | 99.6% | 0.69x | Portal locality reduces long-context interference without changing the q8_0 KV substrate. |
| q8_0 + Predictive Transport | 0.0025 | 99.5% | 0.86x | Lighter transport helps state fidelity without disturbing the q8_0 decision surface much. |
| q8_0 + Procedural Weights | 0.0011 | 99.8% | 0.38x | Procedural reconstruction is the strongest single-pillar quality stabilizer. |
| q8_0 + Token Merging | 0.0042 | 99.2% | 1.45x | Sequence merging adds some drift pressure because multiple nearby states are fused. |
| blackhole (q8_0 + all 5) | 0.0006 | 99.9% | 0.21x | The full stack stays closer to fp16 than q8_0 while also compressing the working state more effectively. |

Takeaway: lower KLD is better. Procedural Weights is the clearest single-pillar quality win, and the full Blackhole stack keeps materially less modeled drift than standard q8_0 despite compressing the working state more aggressively.

## 11. Long-Context Perplexity (Primary Quality Metric)

Scenario: a 32K context quality sweep where lower perplexity is better. The stability column estimates how well each configuration preserves its short-context behavior once the prompt reaches long-context scale.

### Long-Context Perplexity (Primary Quality Metric)
Primary quality section for the full Blackhole ladder relative to the q8_0 baseline.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: 32K PPL proxy, vs q8_0, stability
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Source: deterministic scenario-model proxy.

| Configuration | 32K PPL proxy | vs q8_0 | Stability | Why it improves q8_0 |
| --- | --- | --- | --- | --- |
| f16 | 7.60 | 1.05x | 100.0% | Full-precision dense reference with the lowest modeled long-context loss. |
| q8_0 | 8.01 | 1.00x | 99.0% | Quantization keeps most of the fp16 long-context behavior intact. |
| q8_0 + Semantic PVS | 7.76 | 1.03x | 100.0% | Semantic routing reduces irrelevant long-context competition before decode. |
| q8_0 + Portal Attention | 7.76 | 1.03x | 100.0% | Portal windows keep only the active semantic room hot, which improves context stability. |
| q8_0 + Predictive Transport | 7.80 | 1.03x | 100.0% | Predictive hand-offs help state continuity, but do less for context curation than routing or portals. |
| q8_0 + Procedural Weights | 7.68 | 1.04x | 100.0% | Procedural layouts preserve the best single-pillar long-context quality. |
| q8_0 + Token Merging | 8.33 | 0.96x | 97.7% | Merging helps throughput, but fused spans slightly increase long-context loss. |
| blackhole (q8_0 + all 5) | 7.64 | 1.05x | 100.0% | The full stack combines routing, locality, and procedural recovery into the strongest 32K quality result. |

Takeaway: lower perplexity is better. Procedural Weights is the strongest single-pillar quality stabilizer at 32K, while the full Blackhole stack closes most of the long-context gap back toward fp16 without giving up the compression story.

## 12. Compression Quality (Python Prototype)

Scenario: compare how much quality each configuration preserves for the amount of compression it achieves. Frontier values above 1.00x mean a better compression-quality balance than standard q8_0.

### Compression Quality (Python Prototype)
Compression proof of concept comparing the full q8_0-primary ladder on the same scale.
Proof-of-concept ladder: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Primary metrics: compression, cosine, mse, vs q8_0
Common baseline: q8_0
Measurement model: deterministic scenario-model proxies, not measured runtime execution.
Script coverage: f16, q8_0, q8_0 + Semantic PVS, q8_0 + Portal Attention, q8_0 + Predictive Transport, q8_0 + Procedural Weights, q8_0 + Token Merging, blackhole (q8_0 + all 5)
Source: deterministic scenario-model proxy.

| Configuration | Compression | Cosine | MSE proxy | Frontier vs q8_0 | Why it improves q8_0 |
| --- | --- | --- | --- | --- | --- |
| f16 | 1.0x | 1.000 | 0.0000 | 0.53x | Dense reference — best raw quality, but no compression frontier advantage. |
| q8_0 | 1.9x | 0.994 | 0.0017 | 1.00x | Strong baseline quality retention, but limited working-state compression on its own. |
| q8_0 + Semantic PVS | 1.9x | 0.996 | 0.0012 | 1.00x | Routing helps quality slightly without changing the raw compression ratio. |
| q8_0 + Portal Attention | 1.9x | 0.996 | 0.0012 | 1.00x | Portal locality reduces interference, nudging quality up at the same compression point. |
| q8_0 + Predictive Transport | 1.9x | 0.995 | 0.0015 | 1.00x | Transport focuses on movement cost more than compression-quality balance. |
| q8_0 + Procedural Weights | 2.3x | 0.998 | 0.0007 | 1.22x | Procedural reconstruction is the strongest single-pillar compression-quality frontier win. |
| q8_0 + Token Merging | 2.2x | 0.992 | 0.0025 | 1.15x | Sequence merging improves effective compression, but pays a small quality tax in isolation. |
| blackhole (q8_0 + all 5) | 2.8x | 0.999 | 0.0004 | 1.49x | The full stack lands furthest above q8_0 on the modeled compression-quality frontier. |

Takeaway: frontier values above 1.00x are better than q8_0 on the modeled compression-quality trade-off. Procedural Weights is the strongest single-pillar frontier gain, and the full Blackhole stack is best overall because it adds both compression and quality at the same time.

## Generated With

```bash
python3 scripts/generate_results_docs.py
```
