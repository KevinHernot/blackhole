from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blackhole_core import (
    BlackholePrototype,
    PredictiveTransportCodec,
    activate_portal_context,
    bundle_token_embeddings,
    build_semantic_pvs_index,
    expand_merged_tokens,
    load_tensor_bundle,
    merge_adjacent_tokens,
    proceduralize_matrix,
    relevant_block_recall,
    route_semantic_blocks,
    save_tensor_bundle,
    validate_tensor_bundle,
)


def test_semantic_pvs_routes_relevant_blocks() -> None:
    rng = np.random.default_rng(7)
    dim = 8
    block_size = 8
    clusters = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    tokens = np.vstack(
        [
            center + 0.01 * rng.standard_normal((block_size, dim))
            for center in clusters
        ]
    )
    index = build_semantic_pvs_index(tokens, block_size=block_size)
    query = clusters[1] + 0.01 * rng.standard_normal(dim)
    result = route_semantic_blocks(query, index, top_k=1)
    assert result.selected_blocks.tolist() == [1]
    assert relevant_block_recall([1], result) == 1.0
    assert result.selected_fraction < 0.5


def test_semantic_pvs_rejects_empty_token_matrix() -> None:
    try:
        build_semantic_pvs_index(np.zeros((0, 4)))
    except ValueError as exc:
        assert "at least one token" in str(exc)
    else:
        raise AssertionError("Expected build_semantic_pvs_index() to reject an empty token matrix")


def test_portal_attention_keeps_sinks_current_domain_and_bridge() -> None:
    domains = (
        ["system"] * 4
        + ["code"] * 8
        + ["rag"] * 10
        + ["code"] * 6
    )
    result = activate_portal_context(
        domains,
        "code",
        sink_token_count=4,
        bridge_window=6,
    )
    assert np.all(result.token_mask[:4])
    assert "code" in result.active_domains
    rag_tail = np.arange(len(domains) - 6, len(domains) - 2)
    assert np.any(result.token_mask[rag_tail])


def test_predictive_transport_residual_is_smaller_than_raw_signal() -> None:
    rng = np.random.default_rng(11)
    previous_previous = rng.standard_normal((32, 8))
    previous = previous_previous + 0.05 * rng.standard_normal((32, 8))
    actual = previous + 0.05 * rng.standard_normal((32, 8))

    codec = PredictiveTransportCodec(bit_width=8, velocity_scale=1.0)
    packet, stats = codec.encode(actual, previous, previous_previous)
    reconstructed = codec.decode(packet, previous, previous_previous)

    assert stats.residual_ratio < 1.0
    assert stats.compression_ratio > 1.5
    assert packet.payload_bytes < actual.size * 2
    assert reconstructed.shape == actual.shape
    assert stats.reconstruction.rmse < 0.01


def test_predictive_transport_rejects_bit_width_above_32() -> None:
    try:
        PredictiveTransportCodec(bit_width=40)
    except ValueError as exc:
        assert "<= 32" in str(exc)
    else:
        raise AssertionError("Expected PredictiveTransportCodec() to reject bit widths above 32")


def test_procedural_weights_keep_high_salience_tiles_raw() -> None:
    rng = np.random.default_rng(3)
    matrix = 0.01 * rng.standard_normal((32, 32))
    matrix[:16, :16] += 2.0

    proceduralized = proceduralize_matrix(
        matrix,
        tile_shape=(16, 16),
        keep_high_salience_fraction=0.25,
        basis_rank=4,
    )
    reconstructed = proceduralized.reconstruct()

    assert (0, 0) in proceduralized.raw_tiles
    assert len(proceduralized.procedural_tiles) >= 1
    assert proceduralized.compression_ratio() > 1.0
    assert np.mean((matrix - reconstructed) ** 2) < 0.02


def test_procedural_weights_keep_all_tiles_raw_when_fraction_is_one() -> None:
    matrix = np.arange(16, dtype=float).reshape(4, 4)

    proceduralized = proceduralize_matrix(
        matrix,
        tile_shape=(2, 2),
        keep_high_salience_fraction=1.0,
        basis_rank=1,
    )

    assert len(proceduralized.raw_tiles) == 4
    assert not proceduralized.procedural_tiles


def test_token_merging_merges_redundant_adjacent_tokens() -> None:
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    c = np.array([0.0, 0.0, 1.0, 0.0])
    tokens = np.vstack([a, a, a, b, b, c])

    result = merge_adjacent_tokens(tokens, similarity_threshold=0.9999)
    expanded = expand_merged_tokens(result)

    assert result.merged_tokens.shape[0] == 3
    assert expanded.shape == tokens.shape
    assert result.reduction_fraction > 0.4


def test_token_merging_rejects_non_positive_weights() -> None:
    tokens = np.eye(2)

    try:
        merge_adjacent_tokens(tokens, weights=[1.0, 0.0])
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("Expected merge_adjacent_tokens() to reject zero weights")


def test_blackhole_prototype_composes_context_reduction() -> None:
    rng = np.random.default_rng(19)
    tokens = np.vstack(
        [
            np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (6, 1)),
            np.tile(np.array([[0.0, 1.0, 0.0, 0.0]]), (6, 1)),
            np.tile(np.array([[0.0, 0.0, 1.0, 0.0]]), (6, 1)),
        ]
    )
    tokens = tokens + 0.001 * rng.standard_normal(tokens.shape)
    domains = ["system"] * 2 + ["code"] * 8 + ["rag"] * 4 + ["code"] * 4

    prototype = BlackholePrototype()
    prepared = prototype.prepare_context(tokens, domains)
    active = prototype.active_context(np.array([0.0, 1.0, 0.0, 0.0]), prepared, "code")

    assert prepared.merged_tokens.shape[0] < tokens.shape[0]
    assert active.active_tokens.shape[0] <= prepared.merged_tokens.shape[0]
    assert "code" in active.active_domains


def test_blackhole_prototype_does_not_merge_across_domain_boundaries() -> None:
    tokens = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    domains = ("system", "code")

    prototype = BlackholePrototype()
    prepared = prototype.prepare_context(tokens, domains)

    assert prepared.merged_tokens.shape[0] == 2
    assert prepared.domains == domains


def test_tensor_bundle_round_trip_and_validation(tmp_path) -> None:
    rng = np.random.default_rng(23)
    k_cache = rng.standard_normal((2, 2, 12, 8))
    v_cache = rng.standard_normal((2, 2, 12, 8))
    activations = rng.standard_normal((4, 12, 8))
    domains = ("system", "system") + ("code",) * 6 + ("rag",) * 4

    bundle_path = tmp_path / "bundle.npz"
    from blackhole_core import TensorBundle

    save_tensor_bundle(
        bundle_path,
        TensorBundle(
            k_cache=k_cache,
            v_cache=v_cache,
            activations=activations,
            domains=domains,
        ),
    )
    loaded = load_tensor_bundle(bundle_path)
    embeddings = bundle_token_embeddings(loaded)
    report = validate_tensor_bundle(loaded)

    assert embeddings.shape == (12, 8)
    assert loaded.domains == domains
    assert report["merged_tokens"] <= report["original_tokens"]
    assert report["active_tokens"] <= report["merged_tokens"]
    assert "transport_residual_ratio" in report


def test_tensor_bundle_rejects_legacy_object_domains(tmp_path) -> None:
    legacy_path = tmp_path / "legacy_bundle.npz"
    np.savez(
        legacy_path,
        k_cache=np.ones((2, 2), dtype=float),
        v_cache=np.ones((2, 2), dtype=float),
        domains=np.asarray(["legacy", "legacy"], dtype=object),
    )

    try:
        load_tensor_bundle(legacy_path)
    except ValueError as exc:
        assert "unsupported object array" in str(exc)
    else:
        raise AssertionError("Expected load_tensor_bundle() to reject legacy object-array domains")


def test_tensor_bundle_rejects_domain_length_mismatch(tmp_path) -> None:
    invalid_path = tmp_path / "invalid_bundle.npz"
    np.savez(
        invalid_path,
        k_cache=np.ones((2, 2, 3, 4), dtype=float),
        v_cache=np.ones((2, 2, 3, 4), dtype=float),
        domains=np.asarray(["system", "code"], dtype=str),
    )

    try:
        load_tensor_bundle(invalid_path)
    except ValueError as exc:
        assert "domains must match token dimension" in str(exc)
    else:
        raise AssertionError("Expected load_tensor_bundle() to reject mismatched domain metadata")
