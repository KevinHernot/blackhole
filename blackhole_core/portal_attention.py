from __future__ import annotations

"""Portal Attention prototype for domain-scoped context activation."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import ensure_2d


@dataclass(frozen=True)
class PortalAttentionResult:
    token_mask: np.ndarray
    sink_indices: np.ndarray
    bridge_indices: np.ndarray
    active_domains: tuple[str, ...]


def activate_portal_context(
    domains: Sequence[str],
    current_domain: str,
    *,
    sink_token_count: int = 4,
    bridge_window: int = 32,
    cursor: int | None = None,
    extra_domains: Sequence[str] = (),
) -> PortalAttentionResult:
    domain_array = np.asarray(tuple(domains), dtype=object)
    if domain_array.ndim != 1:
        raise ValueError("domains must be a 1D sequence.")
    if domain_array.size == 0:
        raise ValueError("domains must contain at least one token.")
    if sink_token_count < 0 or bridge_window < 0:
        raise ValueError("sink_token_count and bridge_window must be >= 0.")

    if cursor is None:
        cursor = domain_array.size - 1
    if not 0 <= cursor < domain_array.size:
        raise ValueError(f"cursor must be in [0, {domain_array.size}), got {cursor}.")

    token_mask = domain_array == current_domain
    if extra_domains:
        token_mask |= np.isin(domain_array, np.asarray(tuple(extra_domains), dtype=object))

    sink_indices = np.arange(min(sink_token_count, domain_array.size), dtype=int)
    token_mask[sink_indices] = True

    if bridge_window:
        bridge_start = max(0, cursor - bridge_window + 1)
        bridge_indices = np.arange(bridge_start, cursor + 1, dtype=int)
        token_mask[bridge_indices] = True
    else:
        bridge_indices = np.empty(0, dtype=int)

    active_domains = tuple(dict.fromkeys(domain_array[token_mask].tolist()))
    return PortalAttentionResult(
        token_mask=token_mask,
        sink_indices=sink_indices,
        bridge_indices=bridge_indices,
        active_domains=active_domains,
    )


def gather_portal_tokens(
    token_embeddings: np.ndarray | list[list[float]],
    result: PortalAttentionResult,
) -> np.ndarray:
    token_array = ensure_2d(token_embeddings, name="token_embeddings")
    if token_array.shape[0] != result.token_mask.shape[0]:
        raise ValueError(
            "token_embeddings and token_mask must align: "
            f"{token_array.shape[0]} != {result.token_mask.shape[0]}"
        )
    return token_array[result.token_mask]


__all__ = [
    "PortalAttentionResult",
    "activate_portal_context",
    "gather_portal_tokens",
]
