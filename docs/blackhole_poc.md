# Pillar 1: Semantic PVS

## Scope

This note describes the exact mathematical object implemented today by the Python reference for Semantic PVS in:

- `blackhole/blackhole_core/semantic_pvs.py`
- `blackhole/blackhole_core/stack.py`
- `blackhole/tests/test_blackhole_core_algorithms.py`

These `blackhole_core` files are the source of truth for the mathematics described below.

It also records, only secondarily, the main runtime-side differences visible in:

- `blackhole_runtime/src/blackhole-reference.cpp`

The goal is precision. This is not a conceptual rewrite of the pillar. It is a direct explanation of what the current code actually computes.

Mathematical indexing below is 1-based; Python arrays, half-open slice endpoints, boolean masks, and returned `selected_blocks` ids are 0-based.

## What Semantic PVS does relative to the q8_0 baseline

The q8_0 baseline still keeps the full active KV surface alive. Semantic PVS changes a different control variable: it reduces which semantic blocks are routed into the active decode frontier at all.

In the current Python prototype, Semantic PVS does not:

- learn a router
- train centroids
- use attention scores directly
- preserve individual tokens selectively inside a selected block

Instead, it performs blockwise semantic routing using normalized mean block centroids and cosine similarity against the current query.

## Exact inputs and outputs

### Input state

Semantic PVS operates on a merged token matrix, not on the original token sequence.

Let

- $M \in \mathbb{R}^{m \times d}$ be the merged token matrix
- $b \ge 1$ be the semantic block size
- $q \in \mathbb{R}^d$ be the current query vector

The integrated stack in `stack.py` builds this state as:

1. `merge_adjacent_tokens(...)` produces merged tokens
2. `build_semantic_pvs_index(merged_tokens, block_size=b)` produces the routing index
3. `route_semantic_blocks(query, index, ...)` performs routing

### Index object

`build_semantic_pvs_index(...)` returns a `SemanticPVSIndex` with:

- `block_size`
- `centroids`
- `block_token_ranges`
- `token_count`

If the merged length is $m$, the code partitions the merged rows into consecutive half-open Python slice ranges

- $[s_j, e_j)$ for $j = 1, \dots, r$

where each slice contains at most $b$ merged tokens.

The stored pair $(s_j, e_j)$ is a 0-based Python slice boundary pair. The corresponding mathematical block is

$$B_j = \{s_j + 1, \dots, e_j\} \subseteq \{1, \dots, m\}.$$

### Output object

`route_semantic_blocks(...)` returns a `SemanticPVSResult` with:

- `block_scores`
- `selected_blocks`
- `token_mask`
- `selected_fraction`

So the pillar output is not just a block set. It is a block set plus a token-level active mask induced by those blocks.

Here

$$\text{selected\_fraction} = \frac{|A|}{m} = \frac{1}{m} \sum_{i=1}^m \mathbf{1}\{i \in A\}$$

is the fraction of merged tokens that remain active after routing.

## Mathematical construction

### Step 1: block partition

For each block

- $B_j = \{s_j + 1, \dots, e_j\}$

the code extracts

- $M_j = M[s_j:e_j]$

### Step 2: block centroid

The raw centroid of block $j$ is

$$\mu_j = \frac{1}{|B_j|} \sum_{i \in B_j} M_i$$

The stored block centroid is the L2-normalized version

$$c_j = \frac{\mu_j}{\max(\|\mu_j\|_2, \epsilon)}$$

with $\epsilon = 10^{-12}$.

This normalization comes from `l2_normalize(...)` in `metrics.py`.

### Step 3: query normalization and cosine scoring

The query is normalized the same way:

$$\hat{q} = \frac{q}{\max(\|q\|_2, \epsilon)}$$

Then the score of block $j$ is

$$\sigma_j = \langle c_j, \hat{q} \rangle$$

In code, this is implemented by `pairwise_cosine_scores(query, index.centroids)`.

### Step 4: block selection rule

The selected block indicator is built as the union of up to five rules.

#### Rule 0: direct-API default top-k

If both `top_k` and `similarity_threshold` are omitted, Python first sets

$$\text{top\_k} = \min(4, r)$$

before applying the remaining rules.

This default matters only when `route_semantic_blocks(...)` is called directly. The integrated stack in `stack.py` always supplies both `top_k` and `similarity_threshold`.

#### Rule A: threshold selection

If `similarity_threshold` is $\tau$, mark block $j$ selected when

$$\sigma_j \ge \tau$$

#### Rule B: top-k selection

If `top_k` is $k$, let $\pi$ be the 0-based block-id array returned by `np.argsort(block_scores)`. The implementation reverses that array and keeps its first $k$ entries:

$$\mathcal{J}_{\text{top-k}}^{\text{py}} = \pi[::-1][:k]$$

This is the exact implementation rule. When scores are distinct, it coincides with selecting the $k$ highest-scoring blocks. When scores are tied, the code follows the order returned by `np.argsort(...)`; it does not impose a separate mathematical tie-breaking rule beyond that implementation order.

#### Rule C: preserve-mask forcing

If `force_keep_tokens` is provided as a boolean token mask over merged tokens, then for every block $B_j$,

if any token in $B_j$ is marked `True`, then block $j$ is forced selected.

Formally, if $P \subseteq \{1, \dots, m\}$ is the set of preserved merged-token indices, then

$$B_j \cap P \neq \emptyset \implies j \text{ is selected}$$

In Python, this same object is represented by a 0-based boolean array `force_keep_tokens` of length $m$.

This is the exact mechanism by which Portal Attention influences Semantic PVS in the Python stack.

#### Rule D: nonempty fallback

Let $m_{sel}$ be the minimum number of selected blocks (Python default is $1$).

If after Rules A-C the number of selected blocks is $|\mathcal{J}_{ABC}| < m_{sel}$, the algorithm forces the first $m_{sel}$ entries of `np.argsort(block_scores)[::-1]`:

$$\mathcal{J}_{\text{final}} = \mathcal{J}_{ABC} \cup \mathcal{J}_{\text{fallback}}^{\text{py}}$$

with

$$\mathcal{J}_{\text{fallback}}^{\text{py}} = \operatorname{argsort}(\sigma)[::-1][:m_{sel}]$$

The same implementation-defined tie behavior from Rule B applies here as well.

This guarantees at least one selected block when the index is nonempty.

## Token mask induced by selected blocks

After the block set is chosen, the output token mask is initialized as:

- the preserve mask if one exists
- otherwise all `False`

Then every token inside every selected block is marked active.

Formally, if $J$ is the selected block set, then the final active token set is

$$A = P \cup \left( \bigcup_{j \in J} B_j \right)$$

where $P$ is empty if no preserve mask was supplied.

This means the current implementation is coarse-grained:

- once a block is selected, every token in that block becomes active
- there is no sub-block routing

## Exact Python defaults in the integrated stack

From `BlackholeConfig` in `stack.py`, the integrated defaults are:

- `semantic_block_size = 16`
- `semantic_top_k = 2`
- `semantic_similarity_threshold = 0.10`

Additional stack behavior:

- block size is clipped to the merged length
- `top_k` is clipped to the actual block count
- `Portal Attention` provides `force_keep_tokens`

So the default integrated behavior is:

1. merge tokens
2. build 16-token semantic blocks on the merged sequence
3. select blocks by threshold $0.10$
4. union with top-2 blocks
5. force-select blocks touching the portal-preserve mask
6. if selection were still empty, force the best-scoring block; under the integrated default `semantic_top_k = 2` and a nonempty index, this fallback is redundant

## Exact properties that are true of the code

### Preserve safety

If a merged token is marked in `force_keep_tokens`, it will remain active in the output token mask.

This is exact, because:

1. any block touching a kept token is forced selected
2. the output token mask is the union of the keep mask and selected blocks

### Nonempty routing

If the index contains at least one block, at least one block will be selected.

This is exact because `min_selected` defaults to $1$ and the code forces the top-scoring block if needed.

### Blockwise monotonicity

Holding the index fixed:

- lowering the threshold can only add blocks
- increasing `top_k` can only add blocks

The implementation is monotone because all selection rules are combined by boolean union.

### Coarse block spillover

The implementation can activate semantically irrelevant tokens that share a selected block with a relevant one.

This is not a bug. It is the direct consequence of blockwise routing.

## Complexity

Let:

- $m$ be the merged token count
- $d$ the hidden dimension
- $r = \lceil m / b \rceil$ the number of blocks

Then:

- index build cost is $\mathcal{O}(m d)$
- route scoring cost is $\mathcal{O}(r d)$
- top-k by full sort is $\mathcal{O}(r \log r)$
- token-mask materialization is $\mathcal{O}(m)$

So the current prototype is simple and cheap enough to sit before a more expensive value-fetch stage.

## How this appears in tests

The strongest direct test is `test_semantic_pvs_routes_relevant_blocks()`:

- a synthetic 3-cluster token set is built
- a query near cluster 2 is routed with `top_k=1`
- the selected block is exactly `[1]`, meaning the second block in Python's 0-based block numbering
- recall on the relevant block is $1.0$
- selected fraction is less than $0.5$

So the intended behavior is not just qualitative. The current tests explicitly expect block-level semantic isolation on a clean clustered toy case.

## Interaction with the rest of Blackhole

Semantic PVS never sees original tokens inside the integrated stack. It always sees the output of Token Merging first.

Its dependence graph in Python is:

1. `Token Merging` defines the merged sequence
2. `Portal Attention` defines a preserve mask on that merged sequence
3. `Semantic PVS` routes merged semantic blocks under that preserve constraint

That ordering matters. Mathematically, Semantic PVS is not the first context operator in the stack. It is the final routing operator over the already merged and portal-conditioned view.

## What the current Python implementation does not yet do

It does not:

- learn centroids
- update centroids online
- use query-key attention logits
- route at token granularity
- expose uncertainty estimates
- exploit hierarchical routing

So any theory written for the paper should stay very close to this actual object:

- normalized mean-block centroids
- cosine scores
- threshold plus top-k plus preserve forcing
- nonempty fallback

## Runtime note: important differences in `blackhole_runtime`

The C++ reference runtime implements the same high-level pillar, but not as a bit-for-bit copy of the Python path.

Important differences:

- `Portal Attention` and `Semantic PVS` are fused into one function: `route_tokens(...)`
- if `semantic_top_k == 0`, runtime derives `top_k` from `semantic_keep_fraction`
- runtime can cap selected blocks via `max_active_blocks`
- runtime computes portal forcing and block routing in one pass over merged rows

What remains the same:

- blockwise consecutive partition
- normalized mean block centroids
- cosine scoring
- threshold selection
- top-k selection
- preserve forcing
- nonempty fallback to the best-scoring block

So the mathematical core survives, but the runtime planner adds extra control knobs beyond the Python reference.

# Pillar 2: Portal Attention

## Scope

This note describes the exact Portal Attention object implemented today by:

- `blackhole/blackhole_core/portal_attention.py`
- `blackhole/blackhole_core/stack.py`
- `blackhole/tests/test_blackhole_core_algorithms.py`

These `blackhole_core` files are the source of truth for the mathematics described below.

It contrasts them only secondarily with the current runtime reference in:

- `blackhole_runtime/src/blackhole-reference.cpp`

The goal is to pin down the real math of the current code, not the broader design story.

Indexing below is 0-based throughout, matching Python arrays, boolean masks, and the returned `sink_indices` and `bridge_indices`.

## What Portal Attention does relative to the q8_0 baseline

The q8_0 baseline still treats the available context as one active compressed memory surface. Portal Attention adds a locality prior over that surface.

In the current Python implementation, Portal Attention does not route blocks. It constructs a boolean preserve mask over merged tokens. That mask is then passed into Semantic PVS as `force_keep_tokens`.

So, in the current stack:

- Portal Attention is not the final router
- Semantic PVS is the final router
- Portal Attention biases routing by forcing some merged tokens to remain active

## Exact inputs and outputs

### Inputs

`activate_portal_context(...)` takes:

- `domains`: a length-$m$ sequence of domain labels
- `current_domain`
- `sink_token_count`
- `bridge_window`
- `cursor`
- `extra_domains`

In the integrated stack, `domains` is the merged-domain sequence:

- each merged token inherits the domain of the first original token in its merged span

### Output

The function returns `PortalAttentionResult` with:

- `token_mask`
- `sink_indices`
- `bridge_indices`
- `active_domains`

So the fundamental output is a boolean mask over merged-token coordinates.

## Mathematical construction

Let:

- $\bar{\delta} = (\delta_0, \dots, \delta_{m-1})$ be the merged domain sequence
- $d^*$ be the current domain
- $E$ be the optional set of extra domains
- $s$ be the sink budget
- $w$ be the bridge-window width
- $c$ be the cursor on merged coordinates, with $0 \le c < m$

The preserve set built by the Python code is

$$P = D_{\text{current}} \cup D_{\text{extra}} \cup \Sigma \cup \Gamma$$

where:

- $D_{\text{current}} = \{ i \in \{0, \dots, m - 1\} : \delta_i = d^* \}$
- $D_{\text{extra}} = \{ i \in \{0, \dots, m - 1\} : \delta_i \in E \}$
- $\Sigma = \{ 0, 1, \dots, \min(s, m) - 1 \}$
- $\Gamma = \emptyset$ if $w = 0$, otherwise $\Gamma = \{ \max(0, c - w + 1), \dots, c \}$

The output boolean mask is the indicator of $P$.

## Step-by-step implementation

### Step 1: current-domain activation

The initial mask is, for $i \in \{0, \dots, m - 1\}$,

$$\text{token\_mask}[i] = 1 \iff \delta_i = d^*$$

### Step 2: extra-domain activation

If `extra_domains` is not empty, the mask is expanded by, for $i \in \{0, \dots, m - 1\}$,

$$\delta_i \in E$$

This is implemented by `np.isin(...)`.

### Step 3: sink activation

Let `sink_indices` be the sequence of indices $\{0, 1, \dots, \min(s, m) - 1\}$.

These indices are always turned on.

### Step 4: bridge activation

The code requires the cursor to be a valid merged-token index, so $0 \le c < m$.

If $w > 0$, let

- $\text{bridge\_start} = \max(0, c - w + 1)$
- `bridge_indices` be the sequence of indices $\{\text{bridge\_start}, \dots, c\}$

These indices are always turned on.

If $w = 0$, the bridge set is empty.

### Step 5: active-domain summary

The Python function also returns

- `active_domains = tuple(dict.fromkeys(domain_array[token_mask].tolist()))`

So `active_domains` is the order-preserving deduplicated sequence of domains that survived the mask.

This is not used for routing math itself, but it is part of the observable API.

## Exact Python defaults in the integrated stack

From `BlackholeConfig` in `stack.py`:

- `portal_sink_token_count = 4`
- `portal_bridge_window = 16`

From `BlackholePrototype.active_context(...)`:

- if `current_domain` is omitted, it defaults to the last merged domain
- if `cursor` is omitted, it defaults to the last merged token index

So the default integrated preserve mask is:

- all merged tokens from the most recent domain
- the first 4 merged tokens as sinks
- the last 16 merged tokens as bridge

## Exact properties that are true of the code

### Coverage guarantee

The output mask always contains:

- every token in the current domain
- every token in `extra_domains`
- every sink token
- every bridge token

This is exact because each set is inserted by explicit boolean union.

### No ranking

Portal Attention in Python contains no:

- similarity score
- learned gating score
- top-k
- threshold

It is a deterministic set constructor.

### Domain dependence is external

The current implementation does not infer the active domain.

It assumes `current_domain` is already supplied by upstream logic.

That means the present Python pillar is mathematically a mask-construction operator, not a domain classifier.

### Cursor dependence is explicit

The bridge region is defined relative to the provided cursor. So the preserve mask is not purely a function of domains. It is also a function of decode position.

## Complexity

Let $m$ be the merged-token count.

The implementation is $\mathcal{O}(m)$:

- one pass for `current_domain`
- one pass for optional `extra_domains`
- $\mathcal{O}(\min(s, m))$ sink activation
- $\mathcal{O}(\min(w, c + 1))$ bridge activation
- one pass through the active mask to build `active_domains`

So the sink and bridge activations are still linear-time substeps bounded by $\mathcal{O}(m)$, and the overall complexity remains $\mathcal{O}(m)$.

## How this appears in tests

A direct test is `test_portal_attention_keeps_sinks_current_domain_and_bridge()`:

- the first 4 sink tokens must remain active
- `"code"` must appear in `active_domains`
- a tail slice of `"rag"` tokens near the end must remain active because the bridge window overlaps them

This test verifies the following semantics:

- sinks are absolute
- current-domain activation is absolute
- bridge tokens intentionally keep a small cross-domain boundary active

## Interaction with the rest of Blackhole

In the Python stack, Portal Attention is used as:

`route_semantic_blocks(..., force_keep_tokens=portal.token_mask)`

So if $P$ is the portal preserve set and $J$ is the routed block-id set, and if routed block $j$ covers the 0-based token index set $B_j^{\text{py}}$, the final active token set is

$$A = P \cup \left( \bigcup_{j \in J} B_j^{\text{py}} \right)$$

This means Portal Attention has two roles:

1. direct preservation of some merged tokens
2. indirect forcing of entire semantic blocks to survive through PVS

One preserved token can therefore activate its whole block under downstream PVS.

## What the current Python implementation does not yet do

It does not:

- infer domains from embeddings
- score inter-domain relevance
- compute portals geometrically
- do graded masking
- express relative importance inside the preserve set

So any paper formalization should not overstate it. The current code computes a deterministic boolean preserve set from domains, sinks, and a bridge window.

## Runtime note: important differences in `blackhole_runtime`

The C++ runtime reference currently implements a narrower portal object than the Python reference.

Main differences:

- runtime portal logic is fused into `route_tokens(...)`
- runtime currently supports only:
  - current domain
  - sink tokens
  - bridge tokens
- runtime does not currently expose Python-style `extra_domains`

What remains the same:

- current-domain activation
- sink-prefix activation
- bridge-window activation
- portal output is a boolean mask over merged-token coordinates

So the runtime preserves the core locality idea, but its exposed portal interface is currently smaller than the Python reference API.

# Pillar 3: Predictive Transport

## Scope

This note describes the predictive-transport codec implemented in:

- `blackhole/blackhole_core/predictive_transport.py`
- `blackhole/blackhole_core/stack.py`
- `blackhole/tests/test_blackhole_core_algorithms.py`

These `blackhole_core` files are the source of truth for the mathematics described below.

It compares them only secondarily with the current runtime reference in:

- `blackhole_runtime/src/blackhole-reference.cpp`

## What Predictive Transport does relative to the q8_0 baseline

The q8_0 baseline reduces representation cost on the cache path. Predictive Transport reduces movement cost.

The current Python pillar does not change the model architecture or routing logic. It approximates activation handoff by:

1. predicting the next activation from recent history
2. quantizing only the residual
3. reconstructing the activation from prediction plus quantized residual

So the central object is a residual codec, not a model change. Because the residual quantization is lossy, reconstruction can still change numerical activations and therefore downstream outputs after decode.

## Inputs and outputs

### Inputs

`PredictiveTransportCodec.encode(...)` takes:

- `actual`
- `previous`
- optional `previous_previous`

and codec hyperparameters:

- `bit_width`
- `velocity_scale`
- `raw_bits_per_value`

All three activation inputs are converted by `as_float_array(...)`, i.e. `np.asarray(..., dtype=float)`, before arithmetic is performed.

### Outputs

It returns:

- `QuantizedTransportPacket`
- `PredictiveTransportStats`

The packet contains:

- `quantized_residual`
- `scale`
- `bit_width`

The stats contain:

- `raw_norm`
- `residual_norm`
- `residual_ratio`
- `compression_ratio`
- `reconstruction`

where `reconstruction` is a `ReconstructionStats` bundle from `metrics.py`.

## Mathematical construction

Let:

- $h_t$ be the true activation to be transported
- $h_{t-1}$ the previous activation
- $h_{t-2}$ the previous-previous activation
- $\gamma$ the velocity scale

### Predictor

The predictor is:

- if $h_{t-2}$ is absent:
  - $\hat{h}_t = h_{t-1}$
- otherwise:
  - $\hat{h}_t = h_{t-1} + \gamma (h_{t-1} - h_{t-2})$

This is implemented in `predict_next_activation(...)`.

So the model is a first-order linear extrapolator with optional velocity scaling.

### Residual

The residual is

$$r_t = h_t - \hat{h}_t$$

### Quantizer

Let:

- $b$ be the bit width
- $q_{\max} = 2^{b-1} - 1$
- $\alpha = \max_i |r_{t,i}| / q_{\max}$ if $\max_i |r_{t,i}| > 0$
- otherwise $\alpha = 1.0$

Then the quantized residual is

$$q_i = \text{clip}\left(\text{round}\left(\frac{r_{t,i}}{\alpha}\right), -q_{\max}, q_{\max}\right)$$

with integer dtype:

- `int8` if $b \le 8$
- `int16` if $b \le 16$
- `int32` otherwise

Here `round` is NumPy `np.round`, so halfway cases follow NumPy's ties-to-even behavior. After rounding and clipping, the values are cast with `.astype(...)` into the dtype above.

The current implementation validates

$$2 \le \text{bit\_width} \le 32$$

at codec construction time.

So the scalar quantizer uses the logical bit width $b$ to define $q_{\max}$. However, the stored residual array is not bit-packed to $b$ bits per value. It is stored in a byte-aligned NumPy integer dtype:

- 8 stored bits per value if $2 \le b \le 8$
- 16 stored bits per value if $9 \le b \le 16$
- 32 stored bits per value if $17 \le b \le 32$

This distinction matters for payload accounting.

### Reconstruction

The reconstructed activation is

$$\tilde{h}_t = \hat{h}_t + \alpha q$$

This is exactly what `decode(...)` computes, after converting `quantized_residual` back to floating point with `.astype(float)`.

## Python defaults in the integrated stack

From `BlackholeConfig` in `stack.py`:

- `transport_bit_width = 8`
- `transport_velocity_scale = 1.0`

So the default integrated codec is an 8-bit residual codec with simple linear extrapolation.

This default lies safely inside the parameter regime described above.

## Error properties that follow from the current code

### No clipping saturation beyond the design range

Because $\alpha$ is chosen from the residual max norm, we have

$$\max_i \left| \frac{r_{t,i}}{\alpha} \right| = q_{\max}$$

when the residual is nonzero.

So clipping does not alter any coordinate beyond the rounding effect itself. The remaining approximation comes from scalar rounding plus the implementation-specific rounding rule used by `np.round`.

### Coordinate-wise bound

For every coordinate,

$$|r_{t,i} - \alpha q_i| \le \frac{\alpha}{2}$$

because $|z - \text{round}(z)| \le 1/2$ for any real $z$.

Therefore,

$$|\tilde{h}_{t,i} - h_{t,i}| \le \frac{\alpha}{2}$$

and hence

- $\|\tilde{h}_t - h_t\|_\infty \le \frac{\alpha}{2}$
- $\|\tilde{h}_t - h_t\|_2 \le \sqrt{n} \frac{\alpha}{2}$

for $n$ transported coordinates.

### Zero-residual case

If the residual is identically zero, the Python code sets:

- `scale = 1.0`
- `quantized_residual = 0`

and reconstruction is exact.

This detail matters because some manuscript drafts incorrectly describe the zero case with `scale = 0`. That is not what the Python code does.

## Stats computed by the Python implementation

### Residual ratio

$$\text{residual\_ratio} = \frac{\|r_t\|_2}{\max(\|h_t\|_2, 10^{-12})}$$

This measures how predictable the current activation is from recent history.

### Compression ratio

The packet payload size in Python is

- `quantized_residual.nbytes`
- plus `np.asarray(scale, dtype=np.float64).nbytes`

So Python counts the scale as one 64-bit float, and the residual payload is counted using the actual NumPy storage dtype rather than the logical bit width $b$.

Equivalently,

$$\text{payload\_bytes}_{\text{py}} = \text{quantized\_residual.nbytes} + 8$$

and

$$\text{compressed\_bits}_{\text{py}} = 8 \cdot \text{payload\_bytes}_{\text{py}} = 8 \cdot \text{quantized\_residual.nbytes} + 64.$$

Then

$$\text{compression\_ratio} = \frac{\text{original\_bits}}{\text{compressed\_bits}}$$

with

- $\text{original\_bits} = \text{actual\_array.size} \times \text{raw\_bits\_per\_value}$
- $\text{compressed\_bits} = 8 \cdot \text{packet.payload\_bytes}$
- $\text{raw\_bits\_per\_value} = 16$ by default

### Reconstruction stats

The code reports:

- MSE
- RMSE
- mean cosine similarity
- relative L2 error

via `reconstruction_stats(...)`.

More precisely:

- MSE is the mean squared error over all scalar entries
- RMSE is the square root of that scalar-entry MSE
- relative L2 error is the ratio of flattened L2 norms, using `.ravel()`
- mean cosine similarity is computed rowwise along the last axis; 1D inputs are treated as one row, and tensors with rank $> 2$ are reshaped to `(-1, last_dim)` before averaging the per-row cosine scores

## How this appears in tests

`test_predictive_transport_residual_is_smaller_than_raw_signal()` checks:

- $\text{residual\_ratio} < 1.0$
- $\text{compression\_ratio} > 1.5$
- $\text{packet.payload\_bytes} < \text{actual.size} \times 2$
- reconstruction shape matches
- RMSE is below $0.01$

So the test suite is asserting the intended qualitative behavior:

- prediction makes the residual smaller than the full signal
- the residual packet is meaningfully cheaper than raw FP16 transport
- reconstruction remains close

## Complexity

If the transported tensor has $N$ scalar entries, then encode and decode are both $\mathcal{O}(N)$.

The codec is fully elementwise except for:

- a max-abs reduction
- norm calculations for reporting

## Interaction with the rest of Blackhole

In `BlackholePrototype`, this codec is instantiated once and applied as an independent activation-side transformation.

It does not depend on:

- Semantic PVS scores
- Portal masks
- Procedural weight tiles

except indirectly through the fact that upstream pillars may make the transported state simpler or smaller.

So mathematically, Predictive Transport is separable from the context-routing machinery.

## What the current Python implementation does not yet do

It does not:

- entropy-code the residual integers
- learn the predictor
- predict with multi-step or nonlinear dynamics
- adapt bit-width per channel
- model cross-coordinate covariance

So the current Python object is a simple uniform scalar residual quantizer around a first-order linear predictor, with byte-aligned integer storage and Python-side payload accounting.

## Runtime note: important differences in `blackhole_runtime`

The runtime C++ reference implements the same high-level codec structure:

- same predictor
- same residual definition
- same max-abs scaling idea
- same rounded, clipped uniform quantization principle

But there are two important accounting differences:

1. runtime compression ratio uses
   - $\text{original\_bits} = N \times 16$
   - $\text{compressed\_bits} = N \times \text{bit\_width} + 32$
   so the scale is counted as 32 bits, not Python's float64 payload size
2. runtime reports only RMSE, residual ratio, and compression ratio, not the full Python `ReconstructionStats`

So the predictor/residual/scale structure is aligned at a high level, but the bookkeeping layer is not identical, and this note does not claim byte-for-byte identity of rounding semantics or serialization conventions across Python and C++.

# Pillar 4: Procedural Weights

## Scope

This note describes the exact low-salience tile procedure implemented in:

- `blackhole/blackhole_core/procedural_weights.py`
- `blackhole/blackhole_core/stack.py`
- `blackhole/tests/test_blackhole_core_algorithms.py`

These `blackhole_core` files are the source of truth for the mathematics described below.

It compares them only secondarily with the current runtime reference in:

- `blackhole_runtime/src/blackhole-reference.cpp`

This pillar is the one with the largest current Python/runtime mathematical divergence, so precision matters.

## What Procedural Weights does relative to the q8_0 baseline

The q8_0 baseline reduces KV representation cost. Procedural Weights addresses a different object: weight residency.

The Python prototype does this by splitting a weight matrix into tiles and treating only some tiles as worth storing verbatim. The rest are approximated by:

1. a scalar mean
2. a deterministic seed
3. a coefficient vector in a seeded orthonormal basis over the flattened tile

So the current Python pillar is not arbitrary procedural generation. It is seeded low-dimensional subspace projection around a tile mean.

## Exact inputs and outputs

### Input

`proceduralize_matrix(...)` takes:

- `matrix`
- `tile_shape`
- optional `salience_threshold`
- `keep_high_salience_fraction`
- `basis_rank`
- `base_seed`

### Output

It returns a `ProceduralizedMatrix` with:

- `original_shape`
- `tile_shape`
- `raw_tiles`
- `procedural_tiles`
- `salience_map`

Each procedural tile is stored as a `ProceduralTileSpec` with:

- `seed`
- `shape`
- `mean`
- `coefficients`
- `basis_rank`
- `salience`

Here `basis_rank` is the effective stored rank after clipping by tile size, so it matches `coefficients.size`.

So the basis itself is not stored. It is regenerated from the seed when reconstructing.

## Mathematical construction

Let the full weight matrix be $W \in \mathbb{R}^{m \times n}$.

### Step 1: tiling

Partition $W$ into rectangular tiles $W_s$ according to `tile_shape` $= (r, c)$.

Edge tiles may be smaller than the nominal tile shape.

### Step 2: salience score

For each tile $W_s$, Python computes

$$\text{salience}(W_s) = \frac{\|W_s\|_F}{\sqrt{|W_s|}}$$

where $|W_s|$ is the number of scalar entries in the tile.

This is exactly RMS magnitude:

$$\text{salience}(W_s) = \sqrt{\frac{1}{|W_s|} \sum_{i \in \text{tile}} W_{s,i}^2}$$

So the current salience notion is energy per entry, not Hessian importance or activation sensitivity.

### Step 3: raw vs procedural split

If `salience_threshold` is not provided ($None$), the algorithm keeps the $k = \lceil f \cdot T \rceil$ most energetic tiles raw, where $f$ is `keep_high_salience_fraction` and $T$ is the total number of tiles.

Let the tiles be enumerated in the row-major order produced by `_iter_tile_slices(...)`, with coordinates $\operatorname{coord}_0, \dots, \operatorname{coord}_{T-1}$ and saliences $\sigma_0, \dots, \sigma_{T-1}$. Let

$$\pi = \operatorname{argsort}(-\sigma, \text{ kind = mergesort})$$

be the permutation returned by the stable descending sort used in Python. Then

$$\mathcal{S}_{\text{raw}} = \{ \operatorname{coord}_{\pi_0}, \dots, \operatorname{coord}_{\pi_{k-1}} \}.$$

Implementation details:

- saliences are sorted descending
- sorting uses stable `mergesort`
- ties are therefore broken by the original row-major tile order
- raw tiles are selected by coordinate membership in the top-ranked set

If `salience_threshold` is provided, then a tile is kept raw iff

$$\text{salience}(W_s) > \text{salience\_threshold}$$

So there are two different selection modes:

- top-fraction mode
- absolute-threshold mode

The integrated stack uses top-fraction mode.

### Step 4: procedural fit for one tile

Take one procedural tile $W_s$.

Flatten it:

$$w_s = \text{vec}(W_s) \in \mathbb{R}^p$$

where $p = |W_s|$.

Compute its mean:

$$\mu_s = \text{mean}(w_s)$$

Center it:

$$w_{s,\text{centered}} = w_s - \mu_s \mathbf{1}$$

Now generate a seeded basis.

#### Seeded basis generation

`_random_basis(seed, size, rank)` does:

1. $r_{\text{eff}} = \max(1, \min(\text{rank}, \text{size}))$
2. draw Gaussian matrix $G \in \mathbb{R}^{\text{size} \times r_{\text{eff}}}$
3. compute QR factorization $G = Q R$
4. return $Q^T$

So the returned basis matrix

$$U_s \in \mathbb{R}^{r_{\text{eff}} \times p}$$

has orthonormal rows:

$$U_s U_s^T = I$$

#### Coefficients

The coefficient vector is

$$c_s = U_s w_{s,\text{centered}}$$

This is exactly the orthogonal projection coefficient vector of the centered tile onto the row space of $U_s$.

Because $U_s$ acts on the flattened vector $w_s \in \mathbb{R}^p$, this is an $r_{\text{eff}}$-dimensional subspace approximation in flattened-tile coordinates. After reshaping back to matrix form, it need not correspond to a low matrix-rank approximation of $W_s$.

### Step 5: reconstruction

To reconstruct the tile, Python regenerates the same basis from the stored seed and computes

$$\tilde{w}_s = \mu_s \mathbf{1} + c_s^T U_s$$

equivalently

$$\tilde{w}_s = \mu_s \mathbf{1} + U_s^T c_s$$

and reshapes back to tile shape.

This is the orthogonal projection of the centered tile onto the seeded subspace, shifted back by the mean.

## Exact projection identity

Let

$$P_s = U_s^T U_s$$

Then

$$\tilde{w}_s - \mu_s \mathbf{1} = P_s (w_s - \mu_s \mathbf{1})$$

and therefore

$$\|w_s - \mu_s \mathbf{1}\|_2^2 = \|c_s\|_2^2 + \|w_s - \tilde{w}_s\|_2^2$$

This is the correct mathematical heart of the Python implementation.

## Exact Python defaults in the integrated stack

From `BlackholeConfig` in `stack.py`:

- `procedural_tile_shape = (16, 16)`
- `procedural_keep_high_salience_fraction = 0.25`
- `procedural_basis_rank = 4`

So, by default:

- the matrix is tiled into $16 \times 16$ patches
- the top 25% most salient tiles remain raw
- the remaining 75% are projected onto seeded subspaces of dimension at most 4

## Compression accounting in Python

`ProceduralizedMatrix.compression_ratio(...)` computes:

- $\text{original\_bits} = \text{total\_values} \times \text{original\_bits\_per\_value}$
- default $\text{original\_bits\_per\_value} = 16$

For raw tiles:

- each raw tile costs $\text{tile.size} \times 16$ bits

For each procedural tile:

- $32$ bits for the seed
- $32$ bits for the mean
- $\text{coefficients.size} \times 32$ bits for the coefficients

So the procedural representation cost is

$$64 + 32 \times \text{coefficients.size} = 64 + 32 \times r_{\text{eff}}$$

bits per procedural tile, ignoring tile shape metadata already known from the tiling structure.

## Reconstruction metrics in Python

`procedural_matrix_stats(...)` measures:

- MSE
- RMSE
- mean cosine similarity
- relative L2 error

between the original matrix and the reconstructed matrix.

More precisely, `mean cosine similarity` is computed by `mean_cosine_similarity(...)` from `metrics.py`: for 2D matrices it is the arithmetic mean of rowwise cosine similarities; 1D inputs are treated as a single row; tensors with rank greater than 2 are reshaped to `(-1, last_dim)` before averaging the per-row cosine scores.

## How this appears in tests

`test_procedural_weights_keep_high_salience_tiles_raw()` checks:

- a deliberately energetic $16 \times 16$ tile is kept raw
- at least one other tile becomes procedural
- compression ratio is greater than $1.0$
- mean squared reconstruction error is below $0.02$

`test_procedural_weights_keep_all_tiles_raw_when_fraction_is_one()` checks:

- if `keep_high_salience_fraction` $= 1.0$, all tiles remain raw
- no procedural tiles are created

So the current tests validate:

- the salience ranking logic
- the raw/procedural split
- the basic approximation quality

## Complexity

Let:

- $T$ be the number of tiles
- $p_s$ the size of tile $s$
- $r_s$ the effective rank of tile $s$

Then:

- tiling and salience scoring are $\mathcal{O}(\text{total\_matrix\_size})$
- ranking raw tiles is $\mathcal{O}(T \log T)$
- each procedural fit costs roughly:
  - basis generation
  - QR
  - projection

The heavy part of the Python implementation is the per-tile QR basis construction.

## What the current Python implementation does not yet do

It does not:

- learn salience from model gradients
- condition reconstruction on activations
- learn the basis
- share bases across tiles
- use entropy coding for coefficients
- model expert routing or conditional generation

So the exact current pillar is:

- energy-based tile ranking
- seed-based random orthonormal projection
- mean-plus-seeded-subspace reconstruction

## Runtime note: important differences in `blackhole_runtime`

This is the pillar with the biggest current math mismatch between Python and runtime.

### Python reference

Python uses:

- seeded Gaussian basis
- QR orthonormalization
- exact stored seed
- top-fraction raw tile selection via sorted salience ranking

### Runtime C++ reference

Runtime uses:

- deterministic cosine basis functions `basis_component(index, rank_index, size)`
- no seeded Gaussian QR basis
- salience threshold from a quantile over `procedural_weight_fraction`
- direct RMSE computation over reconstructed tiles

So the runtime does preserve the broad design principle:

- high-salience tiles raw
- low-salience tiles projected into a compact basis representation

But it is not the same mathematical object as the Python reference.

That means the paper should currently treat Python Procedural Weights as the canonical algorithm reference, and the runtime version as a related executable reference approximation rather than as strict parity.

# Pillar 5: Token Merging

## Scope

This note describes the exact token-merging operator implemented in:

- `blackhole/blackhole_core/token_merging.py`
- `blackhole/blackhole_core/stack.py`
- `blackhole/tests/test_blackhole_core_algorithms.py`
- `blackhole/tests/test_token_merging.py`

These `blackhole_core` files are the source of truth for the mathematics described below.

It compares them only secondarily with the current runtime reference in:

- `blackhole_runtime/src/blackhole-reference.cpp`

Repo-level script entry-point smoke tests are discussed below only as secondary evidence. They are not the source of truth for the token-merging mathematics.

## What Token Merging does relative to the q8_0 baseline

The q8_0 baseline keeps one stored vector per token. Token Merging reduces how many vectors exist downstream at all.

The current Python implementation does not solve a global clustering problem. It performs a greedy left-to-right merge of adjacent tokens when the next token is sufficiently similar to the current running weighted mean.

This is a current proof-of-concept compression operator, not a learned compression module or a global optimal sequence compression scheme.

So the exact current pillar is:

- local
- greedy
- order-dependent
- contiguous

## Exact inputs and outputs

### Inputs

`merge_adjacent_tokens(...)` takes:

- `token_embeddings`
- `similarity_threshold`
- `max_group_size`
- optional `weights`

### Outputs

It returns `TokenMergingResult` with:

- `merged_tokens`
- `spans`
- `reduction_fraction`

Each span is a `MergedSpan` with:

- `start`
- `stop`
- `member_count`
- `total_weight`

The spans are half-open intervals `[start, stop)`.

Mathematical token labels below are 1-based. Python spans, masks, and stack indices are 0-based.

## Mathematical construction

Let the input token sequence be

$$X = (x_1, \dots, x_n) \quad \text{with} \quad x_i \in \mathbb{R}^d$$

and let optional strictly positive weights be

$$\omega_1, \dots, \omega_n$$

If no weights are supplied, Python sets all weights to $1$.

The current implementation validates that explicit weights are finite and strictly positive.

So the clean weighted-mean interpretation below is exact for every admissible Python call, including the integrated Blackhole stack where `stack.py` does not pass explicit weights and the code therefore uses unit weights.

### Greedy state

At any point in the scan, the current candidate merge group is a contiguous interval

$$G = \{s, \dots, t\}$$

represented by:

- weighted sum $S_G = \sum_{i \in G} \omega_i x_i$
- total weight $W_G = \sum_{i \in G} \omega_i$

The current representative is

$$m_G = \frac{S_G}{\max(W_G, \epsilon)}$$

with $\epsilon = 10^{-12}$.

When $W_G \ge \epsilon$, this reduces to the usual weighted mean $S_G / W_G$.

### Extension criterion

For the next token $x_j$, the algorithm computes

$$\text{sim} = \cos(m_G, x_j)$$

and extends the current group iff:

- $\text{sim} \ge \text{similarity\_threshold}$
- and the current group size is below `max_group_size` when that bound is enabled

So the decision rule is:

$$\text{extend} \iff \cos(m_G, x_j) \ge \tau \quad \text{and} \quad |G| < g_{\max}$$

if $g_{\max}$ is provided.

### Emission rule

If the extension criterion fails, the algorithm emits the current merged token

$$m_G = \frac{1}{\max(W_G, \epsilon)} \sum_{i \in G} \omega_i x_i$$

stores span metadata for `[s, j)`, and starts a new group at $j$.

After the scan ends, it emits the final group.

## Exact consequences of the code

### Contiguous partition

The resulting spans always form a partition of $[0, n)$ into contiguous nonempty intervals.

This follows from the left-to-right scan and emit-or-extend structure.

### Weighted means in the nondegenerate-weight regime

Each emitted merged token is exactly the weighted mean of its span whenever that span has $\omega(G_k) \ge \epsilon$.

So for every span $G_k$,

$$m_k = \frac{1}{\omega(G_k)} \sum_{i \in G_k} \omega_i x_i$$

where

$$\omega(G_k) = \sum_{i \in G_k} \omega_i$$

In the integrated stack, this always holds because `weights=None` implies $\omega_i = 1$ for all tokens.

### First-moment preservation within each span

For each emitted span with $\omega(G_k) \ge \epsilon$,

$$\sum_{i \in G_k} \omega_i (x_i - m_k) = 0$$

This is the precise structural fact used later in the paper.

Again, in the integrated stack this condition always holds because the weights are all ones.

### Order dependence

Because the algorithm is greedy and only compares each new token to the current running representative, it is not permutation invariant and not globally optimal.

The same multiset of tokens can merge differently under different orders.

### Local similarity, not global cluster quality

The merge decision depends only on:

- the current local group representative
- the next token

It does not optimize a whole-sequence objective such as total reconstruction distortion.

## Exact Python defaults in the integrated stack

From `BlackholeConfig` in `stack.py`:

- `token_similarity_threshold = 0.995`
- `token_merge_max_group_size = 8`

So the default integrated merge operator is:

- very conservative
- limited to short contiguous groups

This matches the intended use: collapse obvious local redundancy, not aggressively rewrite the sequence.

## Stack integration details that matter mathematically

After merging, `stack.py` constructs merged domains as:

`merged_domains = tuple(domains[span.start] for span in merge_result.spans)`

So each merged token inherits the domain of the first token in its span.

In the current integrated stack, merging is performed independently inside each contiguous domain run before those merged domains are constructed.

So, within `BlackholePrototype.prepare_context(...)`, merged spans never cross domain boundaries.

This is mathematically important because:

- `Portal Attention` constructs a deterministic preserve mask over the merged domain sequence.
- `Semantic PVS` performs the final merged-token routing step, with `stack.py` calling `route_semantic_blocks(..., force_keep_tokens=portal.token_mask)`.

Therefore Token Merging changes not only sequence length, but also the coordinate system used for portal preservation and final semantic routing.

## Redefinition of the Context Atom

The most significant mathematical consequence of Token Merging is the **redefinition of the coordinate system** for the rest of the Blackhole stack.

By transforming the original sequence $\mathcal{X} = (x_1, \dots, x_n)$ into a merged sequence $\mathcal{M} = (m_1, \dots, m_m)$, the "atom" of context is no longer a single token embedding but a merged weighted mean.

Subsequent pillars are affected as follows:

- **Pillar 1 (Semantic PVS)**: In the current Python PoC, it performs coarse blockwise centroid routing with threshold, top-k, and fallback logic over sequences of $m_k$, not $x_i$.
- **Pillar 2 (Portal Attention)**: In the current Python PoC, it defines a deterministic domain-based preserve mask over the same $m$-dimensional index space. It is not the final router by itself.
- **Pillar 3 (Predictive Transport)**: In the current Python PoC, it remains a separate first-order linear extrapolator plus uniform residual quantization codec. Its savings can compound with Token Merging system-wide, but `stack.py` does not hard-wire it to the merged coordinate system.

These descriptions are intentionally implementation-specific. They describe the current Python PoC operators, not a broader learned or final-form architecture.

This sequence transformation is one of the mechanisms by which the current Blackhole PoC reduces downstream context volume.

## Expansion map

`expand_merged_tokens(result)` reconstructs a length-$n$ sequence by repeating each merged token `member_count` times.

This is not an inverse operator. It is a broadcast back to original length for inspection and tests.

If span $G_k$ has size $c_k$, expansion returns:

- $m_k$ repeated $c_k$ times

So expansion preserves shape, not original token identity.

## Reduction metric

The reported reduction is

$$\text{reduction\_fraction} = 1 - \frac{m}{n}$$

where:

- $n$ is original token count
- $m$ is merged token count

So this is a pure sequence-length reduction metric.

## How this appears in tests

The direct invariant checks live in `test_blackhole_core_algorithms.py`.

`test_token_merging_merges_redundant_adjacent_tokens()` builds

- `a, a, a, b, b, c`

and checks:

- merged length is $3$
- expanded sequence has the original shape
- reduction fraction is greater than $0.4$

The same file also checks two integration-relevant facts:

- `test_token_merging_rejects_non_positive_weights()` verifies the strict-positive-weight guard
- `test_blackhole_prototype_does_not_merge_across_domain_boundaries()` verifies that integrated merging respects domain runs

`test_token_merging.py` is different in character. It uses deterministic scenario-model comparisons to assert expected system-level effects:

- lower transport volume than the q8_0 baseline
- lower active-token count than the q8_0 baseline
- a retrieval trade-off:
  - penalty in single-needle mode
  - possible bonus in multi-value mode

Those tests are useful for PoC-level behavioral consistency, but they are not direct mathematical proofs of the token-merging operator and they do not by themselves settle the broader thesis empirically.

Separately, `test_script_execution_and_guards.py` contains script smoke tests that check entry points run and print expected headings. Those are guardrail tests for the repo surface, not evidence for token-merging exactness.

So the project already treats Token Merging as a throughput-oriented pillar with explicit retrieval trade-offs.

## Complexity

Let $n$ be the token count and $d$ the embedding dimension.

The Python implementation is a single left-to-right pass:

- each step computes one cosine similarity in $\mathbb{R}^d$
- each token is added exactly once to a running sum

So the merge pass is $\mathcal{O}(n d)$.

## What the current Python implementation does not yet do

It does not:

- perform lookahead search
- split already formed groups
- optimize a global loss
- merge non-adjacent tokens
- use attention-aware reconstruction

So any mathematical statement in the paper should stay close to what is actually true:

- contiguous greedy weighted barycentric merging with cosine thresholding

## Runtime note: important differences in `blackhole_runtime`

The runtime C++ reference implements a very similar token-merging operator, but not an identical one.

What is the same:

- greedy left-to-right scan
- cosine similarity against current group mean
- contiguous spans
- max-group-size limit
- reduction fraction output
 
Main differences:

- runtime currently uses uniform weights only
- runtime stores domain ids directly in the merge result
- runtime interprets `max_group_size == 0` as unlimited

So the runtime preserves the same qualitative operator family, but the Python reference is slightly richer because it supports explicit per-token weights.
