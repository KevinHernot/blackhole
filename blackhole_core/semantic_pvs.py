from __future__ import annotations

"""Semantic PVS routing prototype."""

from dataclasses import dataclass

import numpy as np

from .metrics import ensure_2d, l2_normalize, pairwise_cosine_scores


@dataclass(frozen=True)
class SemanticPVSIndex:
    block_size: int
    centroids: np.ndarray
    block_token_ranges: tuple[tuple[int, int], ...]
    token_count: int

    @property
    def block_count(self) -> int:
        return len(self.block_token_ranges)


@dataclass(frozen=True)
class SemanticPVSResult:
    block_scores: np.ndarray
    selected_blocks: np.ndarray
    token_mask: np.ndarray
    selected_fraction: float


def build_semantic_pvs_index(
    token_embeddings: np.ndarray | list[list[float]],
    block_size: int = 32,
) -> SemanticPVSIndex:
    token_array = ensure_2d(token_embeddings, name="token_embeddings")
    if token_array.shape[0] == 0:
        raise ValueError("token_embeddings must contain at least one token.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")

    centroids: list[np.ndarray] = []
    block_ranges: list[tuple[int, int]] = []
    for start in range(0, token_array.shape[0], block_size):
        stop = min(token_array.shape[0], start + block_size)
        block = token_array[start:stop]
        centroid = l2_normalize(block.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        block_ranges.append((start, stop))

    return SemanticPVSIndex(
        block_size=block_size,
        centroids=np.vstack(centroids),
        block_token_ranges=tuple(block_ranges),
        token_count=token_array.shape[0],
    )


def route_semantic_blocks(
    query: np.ndarray | list[float],
    index: SemanticPVSIndex,
    *,
    top_k: int | None = None,
    similarity_threshold: float | None = None,
    min_selected: int = 1,
    force_keep_tokens: np.ndarray | list[bool] | None = None,
) -> SemanticPVSResult:
    if index.block_count == 0:
        raise ValueError("SemanticPVSIndex must contain at least one block.")

    block_scores = pairwise_cosine_scores(query, index.centroids)
    selected = np.zeros(index.block_count, dtype=bool)

    if similarity_threshold is not None:
        selected |= block_scores >= similarity_threshold

    if top_k is None and similarity_threshold is None:
        top_k = min(4, index.block_count)

    if top_k is not None:
        bounded_top_k = max(0, min(index.block_count, top_k))
        if bounded_top_k:
            top_indices = np.argsort(block_scores)[::-1][:bounded_top_k]
            selected[top_indices] = True

    keep_mask = np.zeros(index.token_count, dtype=bool)
    if force_keep_tokens is not None:
        keep_mask = np.asarray(force_keep_tokens, dtype=bool)
        if keep_mask.shape != (index.token_count,):
            raise ValueError(
                "force_keep_tokens must match the token dimension: "
                f"{keep_mask.shape} != {(index.token_count,)}"
            )
        for block_index, (start, stop) in enumerate(index.block_token_ranges):
            if np.any(keep_mask[start:stop]):
                selected[block_index] = True

    required = max(1, min(index.block_count, min_selected))
    if np.count_nonzero(selected) < required:
        fallback_indices = np.argsort(block_scores)[::-1][:required]
        selected[fallback_indices] = True

    token_mask = keep_mask.copy()
    for block_index in np.flatnonzero(selected):
        start, stop = index.block_token_ranges[block_index]
        token_mask[start:stop] = True

    return SemanticPVSResult(
        block_scores=block_scores,
        selected_blocks=np.flatnonzero(selected),
        token_mask=token_mask,
        selected_fraction=float(np.mean(token_mask)),
    )


def gather_active_tokens(
    token_embeddings: np.ndarray | list[list[float]],
    result: SemanticPVSResult,
) -> np.ndarray:
    token_array = ensure_2d(token_embeddings, name="token_embeddings")
    if token_array.shape[0] != result.token_mask.shape[0]:
        raise ValueError(
            "token_embeddings and token_mask must align: "
            f"{token_array.shape[0]} != {result.token_mask.shape[0]}"
        )
    return token_array[result.token_mask]


def relevant_block_recall(relevant_blocks: list[int] | np.ndarray, result: SemanticPVSResult) -> float:
    relevant = np.asarray(relevant_blocks, dtype=int)
    if relevant.size == 0:
        return 1.0
    hits = np.intersect1d(relevant, result.selected_blocks)
    return float(hits.size / relevant.size)


__all__ = [
    "SemanticPVSIndex",
    "SemanticPVSResult",
    "build_semantic_pvs_index",
    "gather_active_tokens",
    "relevant_block_recall",
    "route_semantic_blocks",
]
