from __future__ import annotations

"""Lightweight benchmark helpers for the Blackhole NumPy prototype."""

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from .portal_attention import activate_portal_context
from .predictive_transport import PredictiveTransportCodec
from .procedural_weights import proceduralize_matrix
from .semantic_pvs import build_semantic_pvs_index, route_semantic_blocks
from .token_merging import merge_adjacent_tokens


@dataclass(frozen=True)
class TimingResult:
    name: str
    mean_seconds: float
    repeat: int


def benchmark_operation(name: str, callback: Callable[[], object], *, repeat: int = 5) -> TimingResult:
    if repeat < 1:
        raise ValueError("repeat must be >= 1.")
    elapsed = []
    for _ in range(repeat):
        start = time.perf_counter()
        callback()
        elapsed.append(time.perf_counter() - start)
    return TimingResult(name=name, mean_seconds=float(np.mean(elapsed)), repeat=repeat)


def benchmark_blackhole_components(
    token_embeddings: np.ndarray,
    domains: list[str],
    query: np.ndarray,
    weight_matrix: np.ndarray,
    activations: np.ndarray,
    *,
    repeat: int = 5,
) -> dict[str, TimingResult]:
    semantic_index = build_semantic_pvs_index(token_embeddings, block_size=min(16, token_embeddings.shape[0]))
    codec = PredictiveTransportCodec()

    return {
        "semantic_pvs": benchmark_operation(
            "semantic_pvs",
            lambda: route_semantic_blocks(query, semantic_index, top_k=min(2, semantic_index.block_count)),
            repeat=repeat,
        ),
        "portal_attention": benchmark_operation(
            "portal_attention",
            lambda: activate_portal_context(domains, domains[-1], bridge_window=min(16, len(domains))),
            repeat=repeat,
        ),
        "predictive_transport": benchmark_operation(
            "predictive_transport",
            lambda: codec.encode(activations[-1], activations[-2], activations[-3]),
            repeat=repeat,
        ),
        "procedural_weights": benchmark_operation(
            "procedural_weights",
            lambda: proceduralize_matrix(weight_matrix, tile_shape=(16, 16)),
            repeat=repeat,
        ),
        "token_merging": benchmark_operation(
            "token_merging",
            lambda: merge_adjacent_tokens(token_embeddings),
            repeat=repeat,
        ),
    }


__all__ = [
    "TimingResult",
    "benchmark_blackhole_components",
    "benchmark_operation",
]
