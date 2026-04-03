from __future__ import annotations

"""Procedural Weights prototype for low-salience tiles."""

from dataclasses import dataclass, field

import numpy as np

from .metrics import ReconstructionStats, ensure_2d, reconstruction_stats


def tile_salience(tile: np.ndarray | list[list[float]]) -> float:
    tile_array = ensure_2d(tile, name="tile")
    return float(np.linalg.norm(tile_array) / np.sqrt(tile_array.size))


def _random_basis(seed: int, size: int, rank: int) -> np.ndarray:
    effective_rank = max(1, min(rank, size))
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((size, effective_rank))
    q, _ = np.linalg.qr(basis)
    return q.T


@dataclass(frozen=True)
class ProceduralTileSpec:
    seed: int
    shape: tuple[int, int]
    mean: float
    coefficients: np.ndarray
    basis_rank: int
    salience: float


@dataclass
class ProceduralizedMatrix:
    original_shape: tuple[int, int]
    tile_shape: tuple[int, int]
    raw_tiles: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    procedural_tiles: dict[tuple[int, int], ProceduralTileSpec] = field(default_factory=dict)
    salience_map: dict[tuple[int, int], float] = field(default_factory=dict)

    def reconstruct(self) -> np.ndarray:
        return reconstruct_procedural_matrix(self)

    def compression_ratio(
        self,
        *,
        original_bits_per_value: int = 16,
        coefficient_bits: int = 32,
    ) -> float:
        original_bits = np.prod(self.original_shape) * original_bits_per_value
        raw_bits = sum(tile.size * original_bits_per_value for tile in self.raw_tiles.values())
        procedural_bits = 0
        for spec in self.procedural_tiles.values():
            procedural_bits += 32  # seed
            procedural_bits += 32  # mean
            procedural_bits += spec.coefficients.size * coefficient_bits
        total_bits = max(raw_bits + procedural_bits, 1)
        return float(original_bits / total_bits)


def fit_procedural_tile(
    tile: np.ndarray | list[list[float]],
    *,
    basis_rank: int = 4,
    seed: int = 0,
) -> ProceduralTileSpec:
    tile_array = ensure_2d(tile, name="tile")
    flat = tile_array.reshape(-1)
    mean = float(np.mean(flat))
    centered = flat - mean
    basis = _random_basis(seed, flat.size, basis_rank)
    coefficients = basis @ centered
    return ProceduralTileSpec(
        seed=seed,
        shape=tile_array.shape,
        mean=mean,
        coefficients=coefficients,
        basis_rank=basis.shape[0],
        salience=tile_salience(tile_array),
    )


def reconstruct_procedural_tile(spec: ProceduralTileSpec) -> np.ndarray:
    basis = _random_basis(spec.seed, int(np.prod(spec.shape)), spec.basis_rank)
    flat = spec.mean + (spec.coefficients @ basis)
    return flat.reshape(spec.shape)


def _iter_tile_slices(shape: tuple[int, int], tile_shape: tuple[int, int]):
    tile_rows, tile_cols = tile_shape
    for row_index, row_start in enumerate(range(0, shape[0], tile_rows)):
        row_stop = min(shape[0], row_start + tile_rows)
        for col_index, col_start in enumerate(range(0, shape[1], tile_cols)):
            col_stop = min(shape[1], col_start + tile_cols)
            yield (row_index, col_index), slice(row_start, row_stop), slice(col_start, col_stop)


def proceduralize_matrix(
    matrix: np.ndarray | list[list[float]],
    *,
    tile_shape: tuple[int, int] = (32, 32),
    salience_threshold: float | None = None,
    keep_high_salience_fraction: float = 0.25,
    basis_rank: int = 4,
    base_seed: int = 0,
) -> ProceduralizedMatrix:
    matrix_array = ensure_2d(matrix, name="matrix")
    if tile_shape[0] < 1 or tile_shape[1] < 1:
        raise ValueError("tile_shape dimensions must be >= 1.")
    if not 0.0 <= keep_high_salience_fraction <= 1.0:
        raise ValueError("keep_high_salience_fraction must be in [0, 1].")

    tile_entries: list[tuple[tuple[int, int], np.ndarray, float]] = []
    for coord, row_slice, col_slice in _iter_tile_slices(matrix_array.shape, tile_shape):
        tile = matrix_array[row_slice, col_slice]
        tile_entries.append((coord, tile.copy(), tile_salience(tile)))

    saliences = np.array([entry[2] for entry in tile_entries], dtype=float)
    raw_tile_coords: set[tuple[int, int]] | None = None
    if salience_threshold is None:
        raw_tile_count = int(np.ceil(keep_high_salience_fraction * len(tile_entries)))
        ranked_indices = np.argsort(-saliences, kind="mergesort") if saliences.size else np.array([], dtype=int)
        raw_tile_coords = {tile_entries[index][0] for index in ranked_indices[:raw_tile_count]}

    proceduralized = ProceduralizedMatrix(
        original_shape=matrix_array.shape,
        tile_shape=tile_shape,
    )

    for offset, (coord, tile, salience) in enumerate(tile_entries):
        proceduralized.salience_map[coord] = salience
        keep_raw = coord in raw_tile_coords if raw_tile_coords is not None else salience > salience_threshold
        if keep_raw:
            proceduralized.raw_tiles[coord] = tile
        else:
            proceduralized.procedural_tiles[coord] = fit_procedural_tile(
                tile,
                basis_rank=basis_rank,
                seed=base_seed + offset,
            )

    return proceduralized


def reconstruct_procedural_matrix(proceduralized: ProceduralizedMatrix) -> np.ndarray:
    reconstructed = np.zeros(proceduralized.original_shape, dtype=float)
    for coord, row_slice, col_slice in _iter_tile_slices(proceduralized.original_shape, proceduralized.tile_shape):
        if coord in proceduralized.raw_tiles:
            tile = proceduralized.raw_tiles[coord]
        else:
            tile = reconstruct_procedural_tile(proceduralized.procedural_tiles[coord])
        reconstructed[row_slice, col_slice] = tile
    return reconstructed


def procedural_matrix_stats(
    original: np.ndarray | list[list[float]],
    proceduralized: ProceduralizedMatrix,
) -> ReconstructionStats:
    return reconstruction_stats(original, proceduralized.reconstruct())


__all__ = [
    "ProceduralTileSpec",
    "ProceduralizedMatrix",
    "fit_procedural_tile",
    "procedural_matrix_stats",
    "proceduralize_matrix",
    "reconstruct_procedural_matrix",
    "reconstruct_procedural_tile",
    "tile_salience",
]
