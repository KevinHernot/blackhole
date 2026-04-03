# Blackhole Package Architecture

`blackhole_core/` is the Python and NumPy prototype surface for Blackhole's five pillars:

- `semantic_pvs.py` implements coarse semantic block routing over merged tokens using normalized mean centroids, cosine scoring, threshold/top-k selection, and preserve-mask forcing.
- `portal_attention.py` implements deterministic domain-scoped context activation with sink and bridge retention.
- `predictive_transport.py` implements first-order residual transport with local prediction, quantized deltas, and an explicit `2 <= bit_width <= 32` contract.
- `procedural_weights.py` implements low-salience tile replacement using deterministic seeded bases.
- `token_merging.py` implements adjacent-token greedy merging and validates finite strictly positive explicit weights.
- `distortion.py` provides explicit distortion and attention-preservation validation.
- `outlier_channels.py` provides an explicit outlier-channel split strategy for tensors and activations.
- `stack.py` composes those modules into an integrated `BlackholePrototype`; in that integrated path, token merging is performed independently inside contiguous domain runs before portal masking and semantic routing.
- `real_model.py` provides tensor-bundle I/O, Transformers hooks, and hardened bundle-to-bundle quality validation with early token-dimension metadata checks.
- `benchmarks.py` provides lightweight v1 timing helpers for the NumPy layer.

This package is the algorithm-side counterpart to the runtime/backend work. The backend/runtime repo currently lives in the sibling `blackhole_runtime` workspace and is the right place to carry forward:

- `llama.cpp` integration and eventual rename
- Metal kernels and graph integration
- CUDA backend work
- MLX-specific execution paths
- upstream coordination and benchmark hardening at the runtime level

That split mirrors the successful TurboQuant workflow:

- algorithm prototyping and validation in Python
- runtime and backend implementation in the `llama.cpp`-oriented repository

In practice, that means this repo should prefer to consume
runtime-produced bundle captures, perplexity traces, and benchmark artifacts
from `blackhole_runtime` rather than rebuilding a second independent runtime
story here.

Current verification status for the Python package surface is tracked in the
generated docs, especially [`result_summary.md`](./result_summary.md), so the
repo does not need to repeat stale test counts by hand here.
