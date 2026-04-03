from __future__ import annotations

"""Token Merging prototype based on adjacent cosine similarity."""

from dataclasses import dataclass

import numpy as np

from .metrics import cosine_similarity, ensure_2d


@dataclass(frozen=True)
class MergedSpan:
    start: int
    stop: int
    member_count: int
    total_weight: float


@dataclass(frozen=True)
class TokenMergingResult:
    merged_tokens: np.ndarray
    spans: tuple[MergedSpan, ...]
    reduction_fraction: float


def merge_adjacent_tokens(
    token_embeddings: np.ndarray | list[list[float]],
    *,
    similarity_threshold: float = 0.995,
    max_group_size: int | None = None,
    weights: np.ndarray | list[float] | None = None,
) -> TokenMergingResult:
    token_array = ensure_2d(token_embeddings, name="token_embeddings")
    if token_array.shape[0] == 0:
        raise ValueError("token_embeddings must contain at least one token.")
    if max_group_size is not None and max_group_size < 1:
        raise ValueError("max_group_size must be >= 1 when provided.")

    if weights is None:
        weight_array = np.ones(token_array.shape[0], dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != (token_array.shape[0],):
            raise ValueError(
                "weights must match the token dimension: "
                f"{weight_array.shape} != {(token_array.shape[0],)}"
            )
        if not np.all(np.isfinite(weight_array)):
            raise ValueError("weights must contain only finite values.")
        if np.any(weight_array <= 0.0):
            raise ValueError("weights must be strictly positive.")

    merged_tokens: list[np.ndarray] = []
    spans: list[MergedSpan] = []

    start = 0
    current_sum = weight_array[0] * token_array[0]
    current_weight = float(weight_array[0])

    for index in range(1, token_array.shape[0]):
        current_mean = current_sum / max(current_weight, 1e-12)
        similarity = float(cosine_similarity(current_mean, token_array[index]))
        group_size = index - start
        can_extend = max_group_size is None or group_size < max_group_size
        if similarity >= similarity_threshold and can_extend:
            current_sum += weight_array[index] * token_array[index]
            current_weight += float(weight_array[index])
            continue

        merged_tokens.append(current_sum / max(current_weight, 1e-12))
        spans.append(
            MergedSpan(
                start=start,
                stop=index,
                member_count=index - start,
                total_weight=current_weight,
            )
        )
        start = index
        current_sum = weight_array[index] * token_array[index]
        current_weight = float(weight_array[index])

    merged_tokens.append(current_sum / max(current_weight, 1e-12))
    spans.append(
        MergedSpan(
            start=start,
            stop=token_array.shape[0],
            member_count=token_array.shape[0] - start,
            total_weight=current_weight,
        )
    )

    merged_array = np.vstack(merged_tokens)
    return TokenMergingResult(
        merged_tokens=merged_array,
        spans=tuple(spans),
        reduction_fraction=1.0 - (merged_array.shape[0] / token_array.shape[0]),
    )


def expand_merged_tokens(result: TokenMergingResult) -> np.ndarray:
    counts = [span.member_count for span in result.spans]
    return np.repeat(result.merged_tokens, counts, axis=0)


__all__ = [
    "MergedSpan",
    "TokenMergingResult",
    "expand_merged_tokens",
    "merge_adjacent_tokens",
]
