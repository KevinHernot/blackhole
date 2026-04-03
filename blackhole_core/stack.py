from __future__ import annotations

"""Integrated Blackhole prototype that composes the five pillars."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .metrics import ensure_2d
from .outlier_channels import OutlierChannelSplit, OutlierChannelStrategy, split_outlier_channels
from .portal_attention import PortalAttentionResult, activate_portal_context
from .predictive_transport import PredictiveTransportCodec, PredictiveTransportStats, QuantizedTransportPacket
from .procedural_weights import ProceduralizedMatrix, proceduralize_matrix
from .semantic_pvs import (
    SemanticPVSIndex,
    SemanticPVSResult,
    build_semantic_pvs_index,
    gather_active_tokens,
    route_semantic_blocks,
)
from .token_merging import MergedSpan, TokenMergingResult, merge_adjacent_tokens


@dataclass(frozen=True)
class BlackholeConfig:
    semantic_block_size: int = 16
    semantic_top_k: int = 2
    semantic_similarity_threshold: float = 0.10
    portal_sink_token_count: int = 4
    portal_bridge_window: int = 16
    token_similarity_threshold: float = 0.995
    token_merge_max_group_size: int = 8
    transport_bit_width: int = 8
    transport_velocity_scale: float = 1.0
    procedural_tile_shape: tuple[int, int] = (16, 16)
    procedural_keep_high_salience_fraction: float = 0.25
    procedural_basis_rank: int = 4
    outlier_score_metric: str = "max_abs"
    outlier_zscore_threshold: float = 2.5
    outlier_max_fraction: float = 0.125
    outlier_min_channels: int = 1


@dataclass(frozen=True)
class PreparedBlackholeContext:
    original_tokens: np.ndarray
    merged_tokens: np.ndarray
    domains: tuple[str, ...]
    merge_result: TokenMergingResult
    semantic_index: SemanticPVSIndex


@dataclass(frozen=True)
class ActiveBlackholeContext:
    active_tokens: np.ndarray
    active_domains: tuple[str, ...]
    portal: PortalAttentionResult
    routing: SemanticPVSResult


def _merge_tokens_with_domain_boundaries(
    token_embeddings: np.ndarray,
    domains: Sequence[str],
    *,
    similarity_threshold: float,
    max_group_size: int | None,
) -> TokenMergingResult:
    if token_embeddings.shape[0] == 0:
        raise ValueError("token_embeddings must contain at least one token.")

    domain_sequence = tuple(domains)
    merged_segments: list[np.ndarray] = []
    spans: list[MergedSpan] = []

    run_start = 0
    while run_start < token_embeddings.shape[0]:
        run_stop = run_start + 1
        while run_stop < token_embeddings.shape[0] and domain_sequence[run_stop] == domain_sequence[run_start]:
            run_stop += 1

        run_result = merge_adjacent_tokens(
            token_embeddings[run_start:run_stop],
            similarity_threshold=similarity_threshold,
            max_group_size=max_group_size,
        )
        merged_segments.append(run_result.merged_tokens)
        spans.extend(
            MergedSpan(
                start=run_start + span.start,
                stop=run_start + span.stop,
                member_count=span.member_count,
                total_weight=span.total_weight,
            )
            for span in run_result.spans
        )
        run_start = run_stop

    merged_tokens = np.vstack(merged_segments)
    return TokenMergingResult(
        merged_tokens=merged_tokens,
        spans=tuple(spans),
        reduction_fraction=1.0 - (merged_tokens.shape[0] / token_embeddings.shape[0]),
    )


class BlackholePrototype:
    def __init__(self, config: BlackholeConfig | None = None) -> None:
        self.config = config or BlackholeConfig()
        self.transport_codec = PredictiveTransportCodec(
            bit_width=self.config.transport_bit_width,
            velocity_scale=self.config.transport_velocity_scale,
        )

    def prepare_context(
        self,
        token_embeddings: np.ndarray | list[list[float]],
        domains: Sequence[str] | None = None,
    ) -> PreparedBlackholeContext:
        token_array = ensure_2d(token_embeddings, name="token_embeddings")
        if domains is None:
            domains = ("default",) * token_array.shape[0]
        if len(domains) != token_array.shape[0]:
            raise ValueError(
                "domains must match token_embeddings length: "
                f"{len(domains)} != {token_array.shape[0]}"
            )

        merge_result = _merge_tokens_with_domain_boundaries(
            token_array,
            domains,
            similarity_threshold=self.config.token_similarity_threshold,
            max_group_size=self.config.token_merge_max_group_size,
        )
        merged_domains = tuple(domains[span.start] for span in merge_result.spans)
        block_size = min(
            max(1, self.config.semantic_block_size),
            max(1, merge_result.merged_tokens.shape[0]),
        )
        semantic_index = build_semantic_pvs_index(merge_result.merged_tokens, block_size=block_size)
        return PreparedBlackholeContext(
            original_tokens=token_array,
            merged_tokens=merge_result.merged_tokens,
            domains=merged_domains,
            merge_result=merge_result,
            semantic_index=semantic_index,
        )

    def active_context(
        self,
        query: np.ndarray | list[float],
        prepared: PreparedBlackholeContext,
        current_domain: str | None = None,
        *,
        cursor: int | None = None,
    ) -> ActiveBlackholeContext:
        if current_domain is None:
            current_domain = prepared.domains[-1]

        effective_cursor = prepared.merged_tokens.shape[0] - 1 if cursor is None else cursor
        portal = activate_portal_context(
            prepared.domains,
            current_domain,
            sink_token_count=self.config.portal_sink_token_count,
            bridge_window=self.config.portal_bridge_window,
            cursor=effective_cursor,
        )
        routing = route_semantic_blocks(
            query,
            prepared.semantic_index,
            top_k=min(self.config.semantic_top_k, prepared.semantic_index.block_count),
            similarity_threshold=self.config.semantic_similarity_threshold,
            force_keep_tokens=portal.token_mask,
        )
        active_tokens = gather_active_tokens(prepared.merged_tokens, routing)
        active_domains_array = np.asarray(prepared.domains, dtype=object)[routing.token_mask]
        return ActiveBlackholeContext(
            active_tokens=active_tokens,
            active_domains=tuple(active_domains_array.tolist()),
            portal=portal,
            routing=routing,
        )

    def proceduralize_weights(self, weight_matrix: np.ndarray | list[list[float]]) -> ProceduralizedMatrix:
        return proceduralize_matrix(
            weight_matrix,
            tile_shape=self.config.procedural_tile_shape,
            keep_high_salience_fraction=self.config.procedural_keep_high_salience_fraction,
            basis_rank=self.config.procedural_basis_rank,
        )

    def outlier_channel_strategy(self) -> OutlierChannelStrategy:
        return OutlierChannelStrategy(
            score_metric=self.config.outlier_score_metric,
            zscore_threshold=self.config.outlier_zscore_threshold,
            max_outlier_fraction=self.config.outlier_max_fraction,
            min_outlier_channels=self.config.outlier_min_channels,
        )

    def split_outlier_channels(self, values: np.ndarray | list[list[float]]) -> OutlierChannelSplit:
        return split_outlier_channels(values, strategy=self.outlier_channel_strategy())

    def encode_transport(
        self,
        actual: np.ndarray | list[float],
        previous: np.ndarray | list[float],
        previous_previous: np.ndarray | list[float] | None = None,
    ) -> tuple[QuantizedTransportPacket, PredictiveTransportStats]:
        return self.transport_codec.encode(actual, previous, previous_previous)

    def decode_transport(
        self,
        packet: QuantizedTransportPacket,
        previous: np.ndarray | list[float],
        previous_previous: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        return self.transport_codec.decode(packet, previous, previous_previous)


__all__ = [
    "ActiveBlackholeContext",
    "BlackholeConfig",
    "BlackholePrototype",
    "PreparedBlackholeContext",
]
