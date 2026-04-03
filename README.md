# Blackhole

![Blackhole logo](./Blackhole.jpeg)

Blackhole is the proof-of-concept layer for five additive ideas:

- Semantic PVS
- Portal Attention
- Predictive Transport
- Procedural Weights
- Token Merging

The repository currently focuses on Python scripts.

Those scripts report **deterministic scenario-model proxies**, not measured execution from a native Blackhole runtime.

The `blackhole_core/` Python package now also includes real NumPy prototype modules for the five pillars, while the runtime/backend track is the sibling `blackhole_runtime` workspace that is carrying the executable runtime path.

The repo now also has the first measured-eval plumbing beyond pure proxies:

- bundle-backed measured-quality fixture artifacts under `evals/`
- a manifest-driven artifact builder for saved bundle/context captures
- an optional reference-side measured-model runner in `scripts/run_measured_model_eval.py`
- a runtime-capture contract and candidate-manifest initializer for the sibling `blackhole_runtime` workspace

That measured-eval plumbing is meant to meet `blackhole_runtime` halfway: this repo should be the algorithm, analysis and artifact-validation side, while real runtime-produced captures and benchmark traces can come from the sibling runtime workspace.

Blackhole still reads best as a stack of additive improvements on top of standard `q8_0`: the routing and locality pillars move the speed and retrieval story, while the quality sections now declare their source explicitly as proxy scenario-model outputs.

## Key Metrics of the python proof-of-concept benchmark — Full Blackhole vs q8_0

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

> **📊 Full results are in [`docs/results.md`](./docs/results.md)** — detailed tables, heatmaps and analysis.
> **💡 New: read the [accessible result summary](./docs/result_summary.md)** for a plain-language explanation of why these benchmarks matter.
> **🗂️ Docs index: [`docs/readme.md`](./docs/readme.md)** — overview of generated artifacts and where each document fits.

## Origin story: from video games to LLMs

Blackhole didn't start with a machine-learning paper. It started with video games.

The core insight came from studying how 1990s and 2000s game engines solved a problem that is structurally identical to the one modern LLM inference faces: **the memory wall**. Early 3D engines — Doom, Quake, Minecraft — ran on hardware with a few megabytes of RAM and CPUs under 100 MHz. They couldn't afford to render what the player couldn't see, move data the GPU didn't need, or store geometry that could be generated on the fly. Survival meant developing an entire science of *algorithmic illusion*: trading mathematical perfection for perceptual equivalence.

| Modern LLM technique | 90s game equivalent | Shared principle |
| --- | --- | --- |
| Walsh-Hadamard decorrelation | DCT in JPEG/MPEG | Transform to frequency domain, discard what doesn't matter |
| Lloyd-Max codebook (VQ) | 256-color indexed palette | Replace raw data with indices into a small dictionary |
| Norm + quantized unit vector | Quake's 162-entry normal table | Decouple magnitude from direction, look up instead of compute |
| 128-element block quantization | S3 Texture Compression (S3TC) | Compress in local blocks to preserve local variance |
| ADC lookup tables | Precomputed diffuse lighting LUTs | Trade real-time math for instant memory lookups |
| Per-block scale factors | ADPCM audio compression | One high-precision "volume knob" per block of cheap residuals |

If the already-shipped optimizations were rediscoveries of 90s rendering tricks, what about the tricks the AI industry **hadn't** ported yet?

We went through the full catalog of real-time rendering techniques — frustum culling, occlusion culling, portal rendering, dead reckoning, procedural generation, greedy meshing, level-of-detail hierarchies — and asked: *what would each of these look like inside a transformer's attention mechanism?*

The answers became the five Blackhole pillars:

1. **Semantic PVS** came from **Potentially Visible Sets and occlusion culling**. In Quake and Doom, the engine precomputed which rooms were visible from each position and skipped everything else. Translated to LLMs: a lightweight routing layer estimates which KV blocks are relevant to the current query and the engine skips loading the rest from memory entirely. **Don't render what the player can't see and don't attend to what the query doesn't need.**

2. **Portal Attention** came from **portal rendering**. Games like Duke Nukem 3D and Descent split worlds into rooms connected by portal polygons. The engine only rendered what was visible through the current doorway. Translated to LLMs: partition the context into semantic domains (code, instructions, retrieval chunks), keep only the active domain plus a small bridge window live and gate out everything else. **Don't load the entire map when you're standing in one room.**

3. **Predictive Transport** came from **dead reckoning in multiplayer games**. Networked games like Quake III don't send full player coordinates every frame. Instead, the client predicts the next position locally and the server only transmits a correction delta when reality diverges. Translated to LLMs: in distributed pipeline-parallel inference, instead of shipping full activation tensors between devices, predict the next state locally and transmit only the residual error. **Don't send the whole packet when a small correction will do.**

4. **Procedural Weights** came from **procedural generation**. Games like No Man's Sky and Elite Dangerous generate entire galaxies from tiny seeds, trading storage for compute. Translated to LLMs: low-salience weight tiles don't need to be stored and fetched from memory at all. A compact seed plus a fast PRNG can regenerate them on the fly inside the compute units, bypassing the memory bus entirely. **Don't store what you can cheaply recompute.**

5. **Token Merging** came from **greedy meshing in Minecraft**. Voxel engines don't render a separate polygon for every visible face of every block. Instead, they merge adjacent identical faces into single large rectangles, drastically cutting draw calls. Translated to LLMs: adjacent tokens with near-identical KV representations are fused into single "merged" tokens, shrinking the sequence dimension itself. **Don't process a hundred redundant faces when one rectangle will do.**

The Minecraft connection ran especially deep. The research into CraftGPT (a working LLM built entirely in Minecraft's Redstone circuitry) made the physical nature of the problem viscerally clear. In Redstone, every dot product is a literal physical journey across digital distances, governed by the game's 20-tick-per-second clock. It strips away all abstraction and reveals that **inference is fundamentally a routing problem**: how much data must physically move through the system per generation step. That insight (the fact that the bottleneck is *data movement*, not *computation*) is what makes game-engine thinking the right lens.

## The five pillars of Blackhole

Blackhole is thus inspired by the same constraint that shaped old game engines and now shapes long-context LLM inference: once the system hits the **memory wall**, the winning move is often not to compute harder, but to move less data, keep less state active and avoid work that is no longer visible to the current task.

These five pillars are direct translations of game-engine optimization paradigms into the transformer attention mechanism, benchmarked additively on top of `q8_0`.

| Pillar | Game / rendering analogy | Blackhole idea | Main bottleneck it targets |
| --- | --- | --- | --- |
| Semantic PVS | Potentially Visible Sets, occlusion culling | Only route semantically relevant regions into the active attention frontier | Over-scanning irrelevant KV/cache blocks |
| Portal Attention | Portal rendering, room-to-room visibility | Keep only the current domain, bridge tokens and sinks active | Oversized prefill / active-window cost |
| Predictive Transport | Dead reckoning, delta updates in multiplayer engines | Ship corrections and reconstruct the next state instead of copying full payloads | Layer-to-layer or device-to-device transfer bandwidth |
| Procedural Weights | Procedural terrain, seed-based asset generation | Reconstruct low-salience weights from compact seeds or generators | Weight storage and memory-fetch pressure |
| Token Merging | Greedy Meshing, geometry simplification | Fuse redundant adjacent tokens in the sequence dimension | Attention matrix size and transport payload |

### Semantic PVS

Semantic PVS is Blackhole's **macro-routing** pillar. The idea comes from Potentially Visible Set compilation and occlusion culling in game engines: if a room or sector is not visible from the player’s current position, the engine should not pay to render it.

Translated to LLM inference, that means the model should not treat every context region as equally alive for every decode step. A lightweight semantic routing layer can estimate which blocks are actually relevant to the current query and shrink the search frontier before the expensive value path begins. In practical terms, Semantic PVS is about spending bandwidth only on the blocks that are likely to matter, rather than dragging the entire routed history through attention every time.

### Portal Attention

Portal Attention is the **locality and boundary-control** pillar. Its inspiration comes from portal rendering in games, where worlds are split into rooms or zones and the engine only renders what is visible through the current doorway, window, or connection.

For Blackhole, that becomes a context-bank idea: keep the current domain active, preserve a small bridge window for continuity and avoid flooding the active KV working set with unrelated context from distant domains. This is especially useful when prompts move between different semantic regions such as instructions, code, retrieval snippets and synthesis. Instead of one giant always-hot context, Portal Attention tries to keep only the relevant room lit.

### Predictive Transport

Predictive Transport is the **handoff-optimization** pillar. The closest systems analogy is dead reckoning in networked games: rather than transmitting every full state update, predict the next state locally and send only the correction when reality diverges from the prediction.

In an inference stack, adjacent layers, experts, or devices often exchange states that are highly correlated from one step to the next. Blackhole explores whether that handoff can be made cheaper by transporting deltas or reconstruction signals instead of full activations every time. The goal is not to change the reasoning path itself, but to reduce the bandwidth cost of moving the surviving state through the stack.

### Procedural Weights

Procedural Weights is the **model-parameter compression** pillar. The analogy is **procedural generation** from games (like in *No Man's Sky* or Minecraft terrain): instead of storing every unique rock and tree on disk, the engine stores the mathematical rules and seeds to generate them on the fly.

In large language models, weight matrices often contain vast regions of low-salience parameters that contribute little to the final output but consume massive amounts of memory and bandwidth. Procedural Weights identifies these "low-impact" tiles and replaces their raw values with compact procedural specifications—a seed, a mean and a few coefficients. At inference time, these tiles are reconstructed from their "DNA" rather than being dragged from memory, significantly reducing the storage footprint and the I/O bottleneck without sacrificing the model's reasoning capabilities.

### Token Merging

Token Merging is the **sequence-dimension compression** pillar. The analogy is **Greedy Meshing** from 3D graphics (like in Minecraft or voxel engines): instead of rendering six faces for every tiny cube, the engine merges adjacent faces of the same material into a single large rectangle to reduce the number of polygons.

In LLMs, nearby tokens often represent redundant semantic concepts or repeated patterns. Instead of paying to attend to every individual token, Token Merging fuses redundant adjacent tokens in the KV cache into a single "merged" token. This does not just shrink the bit-depth of the data; it shrinks the length of the data itself, drastically reducing the quadratic complexity of the attention matrix and the volume of state transported between layers.

### Why the five pillars fit together

These pillars are meant to be **additive, not redundant**:

- **Semantic PVS** decides which regions deserve attention at all.
- **Portal Attention** decides which local window should stay active right now.
- **Predictive Transport** reduces the cost of moving the surviving activations forward.
- **Procedural Weights** reduces the cost of storing and fetching the model substrate itself.
- **Token Merging** reduces the number of tokens the system must process by fusing redundancy.

Together they attack five different forms of the same underlying enemy: unnecessary bytes moving through the system.

## Prior art: the five pillars in the literature

The five pillars have already been researched — individually, and sometimes under different names. What follows is an honest mapping of each pillar against existing published work, along with what we believe remains genuinely novel in each case.

### Semantic PVS — already explored, and quite strongly

RetrievalAttention replaces dense long-context access with vector search inside the KV cache and reports needing only about 1–3 % of the data consulted; Squeezed Attention compresses a fixed context offline via clustering and then only consults keys deemed important; H2O, LazyLLM and DynamicKV select or retain only the most useful tokens/KV entries. So the "what to look at" question already exists clearly as semantic/dynamic context culling. The still-novel angle, in our view, is a semantic PVS that is genuinely **pre-computed and reusable** per state, tool, corpus or tenant — rather than a selection that is mostly recomputed on the fly.

### Portal Attention — already explored as well

Routing Transformer already does content-based sparse routing via online k-means; NSA combines coarse-grain compression with fine-grained selection; DHSA segments the context into variable-length chunks and only computes attention on relevant chunk pairs; Squeezed Attention also has a hierarchical variant. So the "how to access it" question via coarse-to-fine routing is well represented in the literature. What remains less explored, in our opinion, is the true **"portal graph"** version: an explicit, persistent graph of memory regions / tools / experts with pre-computed adjacency and admissibility — the way a game engine builds its portal topology.

### Predictive Transport — already explored, very concretely at the systems level

SpeCache speculatively predicts which KV pairs will be consulted at the next step and prefetches them; PRESERVE prefetches weights and KV during communication phases; Strata organises a hierarchical context cache with GPU-assisted I/O and cache-aware scheduling; NVIDIA Dynamo documents non-blocking KV transfer directly from VRAM to VRAM in disaggregated serving. So the "when to move data" question is already an active serving topic. Where room remains is in a **semantic predictive transport** guided by agent trajectory, tool plan, or expert route — not merely by cache and I/O heuristics.

### Procedural Weights — already explored, but the most fragmented of the five

LoRA-Flow learns dynamic fusion weights between LoRAs; CoDA adds conditional adapters with sparse activation and reports inference gains; SHINE generates LoRAs from context in a single forward pass; DynMoE and "Harder Task Needs More Experts" vary the number of activated experts; Dr.LLM dynamically routes layers via skip / execute / repeat. So the "with what capacity to compute" idea is well present, but scattered across adaptation, conditional computation, MoE and layer routing. In the papers we have verified, it is not yet a **unified serving primitive** — it is rather a set of specialised mechanisms.

### Token Merging / Greedy Meshing — already explored, but mainly in vision/MLLM; in text AR it is more emergent

ToMe established token merging in vision Transformers; AIM and AdaptMerge do inference-time merging/pruning for MLLMs; QuickMerge++ explicitly pushes token merging toward autoregressive next-token prediction on text/image/video. So the "don't keep every primitive independent" idea already exists. The genuinely open space, in our view, is **structure-aware textual greedy meshing**: fusing adjacent spans/tokens while preserving causality, syntax and provenance — not just embedding similarity.

---

## Current Python PoC behavior

The current NumPy proof of concept is intentionally narrower and more explicit than the higher-level idea story above.

- `stack.py` composes the integrated context path in this order: Token Merging, then Portal Attention, then Semantic PVS.
- In the integrated stack, Token Merging is applied independently inside each contiguous domain run, so merged spans never cross domain boundaries.
- `portal_attention.py` currently builds a deterministic preserve mask over merged-token coordinates from the current domain, sink tokens, bridge tokens and optional extra domains.
- `semantic_pvs.py` currently routes consecutive merged-token blocks using normalized mean block centroids, cosine scoring, threshold and top-k selection and portal-driven preserve forcing.
- `predictive_transport.py` currently exposes a first-order residual codec and validates `2 <= bit_width <= 32`.
- `token_merging.py` currently exposes greedy adjacent cosine merging and validates that explicit weights are finite and strictly positive.
- `real_model.py` now validates tensor-bundle token metadata early, including `domains` length versus the token dimension implied by the KV tensors.

So the current Python PoC should be read as a concrete, testable reference implementation of the five pillars, not just as a conceptual translation from game-engine analogies.

## Canonical Blackhole configurations

All user-facing proof-of-concept output is restricted to exactly these eight labels:

- `f16`
- `q8_0`
- `q8_0 + Semantic PVS`
- `q8_0 + Portal Attention`
- `q8_0 + Predictive Transport`
- `q8_0 + Procedural Weights`
- `q8_0 + Token Merging`
- `blackhole (q8_0 + all 5)`

The common reference point is always **standard `q8_0`**.

## Why the PoC uses `q8_0` as its base

The Python proof of concept is now deliberately q8_0-primary.

That makes the story cleaner:

- `f16` remains the high-fidelity reference
- `q8_0` is the common compressed baseline
- every Blackhole row isolates the effect of one pillar, or the full stack, on top of the same baseline

This keeps the PoC focused on Blackhole itself rather than on the behavior of an external compression family. The paper and runtime track can still study more aggressive cache regimes separately, but this repository's proof-of-concept methodology now attributes gains directly against `q8_0`.

## What the repo is doing today

This repo is **not** pretending to ship a native Blackhole runtime yet.

Instead, the Python scripts are honest about their role:

- they are **proofs of concept**
- they run the **same eight-configuration ladder everywhere**
- they explain how each Blackhole pillar improves on `q8_0`

## Result sections

The shared comparison catalog names these sections:

- `Top-of-Tree Results`
- `Prefill Context Scaling (Verified 2K-32K)`
- `Decode Speed — MoE`
- `NIAH Retrieval`
- `KL Divergence vs f16`
- `Decode Speed — Dense`
- `Long-Context Perplexity (Primary Quality Metric)`
- `Compression Quality (Python Prototype)`
- `Speed Optimization Journey`
- `Sequence Compression — Greedy Meshing`

Each section uses the same nine-config ladder listed above.
All ten catalog entries now have standalone script-backed coverage in `scripts/` and rendered coverage in `docs/results.md`.

## Project structure

```text
blackhole/
├── blackhole_core/
│   ├── __init__.py
│   ├── benchmarks.py
│   ├── comparison_profiles.py
│   ├── distortion.py
│   ├── evidence_tiers.py
│   ├── measured_quality.py
│   ├── metrics.py
│   ├── outlier_channels.py
│   ├── portal_attention.py
│   ├── predictive_transport.py
│   ├── procedural_weights.py
│   ├── real_model.py
│   ├── run_manifest.py
│   ├── runtime_capture_contract.py
│   ├── scenario_model.py
│   ├── semantic_pvs.py
│   ├── stack.py
│   └── token_merging.py
├── README.md
├── pyproject.toml
├── docs/
│   ├── package_architecture.md
│   ├── readme.md
│   ├── result_summary.md
│   └── results.md
├── scripts/
│   ├── _comparison_profiles.py
│   ├── _scenario_model.py
│   ├── README.md
│   ├── compression_quality.py
│   ├── generate_results_docs.py
│   ├── kl_divergence_vs_f16.py
│   ├── long_context_perplexity.py
│   ├── measure_skip_rate.py
│   ├── niah_pvs_routing.py
│   ├── niah_test.py
│   ├── portal_attention.py
│   ├── predictive_transport.py
│   ├── run_measured_model_eval.py
│   ├── semantic_pvs_routing.py
│   ├── token_merging_poc.py
│   └── unified_poc.py
└── tests/
    ├── test_blackhole_core_algorithms.py
    ├── test_blackhole_hardening.py
    ├── test_comparison_profiles.py
    ├── test_generate_results_docs.py
    ├── test_measured_model_workflow.py
    ├── test_measured_quality.py
    ├── test_niah_configurations.py
    ├── test_script_execution_and_guards.py
    └── test_token_merging.py
```

## Quick start

Install the local package:

```bash
python3 -m pip install -e .
```

Run the unified Blackhole summary:

```bash
python3 scripts/unified_poc.py
```

Run the NIAH proof of concept:

```bash
python3 scripts/niah_test.py
```

Refresh the generated docs and manifest:

```bash
python3 scripts/generate_results_docs.py --pytest-status "<latest pytest summary>"
```

`docs/results.md` and `docs/result_summary.md` are now the UTC-stamped source of truth for:

- the latest verification status
- the latest NIAH artifact paths
- whether quality sections are still `proxy` or have been promoted to artifact-backed evidence

## Results at a glance

The q8_0-primary PoC is meant to answer one practical question: how much do the
five Blackhole pillars buy us when we start from a stable compressed baseline
instead of a full-precision reference.

The current generated docs focus on:

- prefill and active-KV reductions relative to `q8_0`
- decode and transport proxy gains relative to `q8_0`
- retrieval improvements across the full NIAH ladder
- quality drift, long-context perplexity and compression frontier deltas relative to `q8_0`

For the latest numbers, read the generated reports rather than this README:

> **📊 Full results → [`docs/results.md`](./docs/results.md)** | **💡 Accessible summary → [`docs/result_summary.md`](./docs/result_summary.md)**

## Where to look next

- **[`docs/result_summary.md`](./docs/result_summary.md)** — plain-language explanation of benchmark implications.
- **[`docs/results.md`](./docs/results.md)** — complete proof-of-concept results with all tables, heatmaps and cross-pillar analysis.
- **[`docs/readme.md`](./docs/readme.md)** — documentation index with the latest generated artifact paths.
- `scripts/README.md` explains the proof-of-concept role of each script.
- `blackhole_core/comparison_profiles.py` is the source of truth for labels and ordering.
- `blackhole_core/scenario_model.py` is the shared source of truth for proof-of-concept mechanics and proxy numbers.
- `blackhole_core/semantic_pvs.py`, `blackhole_core/portal_attention.py`, `blackhole_core/predictive_transport.py`, `blackhole_core/procedural_weights.py` and `blackhole_core/token_merging.py` are the real NumPy prototype implementations for the five pillars.
- `blackhole_core/distortion.py`, `blackhole_core/outlier_channels.py` and `blackhole_core/real_model.py` harden the Python layer with explicit distortion checks, outlier handling and bundle-to-bundle quality validation.
- `blackhole_core/stack.py` composes those pillar modules into an integrated `BlackholePrototype`.
- [`docs/package_architecture.md`](./docs/package_architecture.md) explains the split between the Python algorithm package and the sibling runtime/backend repository.
- `scripts/_comparison_profiles.py` and `scripts/_scenario_model.py` remain as compatibility shims for the script entry points.
- `tests/test_blackhole_core_algorithms.py`, `tests/test_blackhole_hardening.py`, `tests/test_script_execution_and_guards.py`, `tests/test_comparison_profiles.py` and `tests/test_niah_configurations.py` cover the package and proof-of-concept contract.
