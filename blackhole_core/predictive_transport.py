from __future__ import annotations

"""Predictive Transport prototype for residual-based activation transfer."""

from dataclasses import dataclass

import numpy as np

from .metrics import ReconstructionStats, as_float_array, reconstruction_stats


def predict_next_activation(
    previous: np.ndarray | list[float],
    previous_previous: np.ndarray | list[float] | None = None,
    *,
    velocity_scale: float = 1.0,
) -> np.ndarray:
    previous_array = as_float_array(previous, name="previous")
    if previous_previous is None:
        return previous_array.copy()

    previous_previous_array = as_float_array(previous_previous, name="previous_previous")
    if previous_previous_array.shape != previous_array.shape:
        raise ValueError(
            "previous and previous_previous must share the same shape: "
            f"{previous_array.shape} != {previous_previous_array.shape}"
        )
    velocity = previous_array - previous_previous_array
    return previous_array + velocity_scale * velocity


@dataclass(frozen=True)
class QuantizedTransportPacket:
    quantized_residual: np.ndarray
    scale: float
    bit_width: int

    @property
    def payload_bytes(self) -> int:
        return int(self.quantized_residual.nbytes + np.asarray(self.scale, dtype=np.float64).nbytes)


@dataclass(frozen=True)
class PredictiveTransportStats:
    raw_norm: float
    residual_norm: float
    residual_ratio: float
    compression_ratio: float
    reconstruction: ReconstructionStats


def _quantized_dtype(bit_width: int) -> np.dtype:
    if bit_width <= 8:
        return np.int8
    if bit_width <= 16:
        return np.int16
    return np.int32


class PredictiveTransportCodec:
    def __init__(
        self,
        *,
        bit_width: int = 8,
        velocity_scale: float = 1.0,
        raw_bits_per_value: int = 16,
    ) -> None:
        if bit_width < 2:
            raise ValueError("bit_width must be >= 2.")
        if bit_width > 32:
            raise ValueError("bit_width must be <= 32.")
        self.bit_width = bit_width
        self.velocity_scale = velocity_scale
        self.raw_bits_per_value = raw_bits_per_value

    def encode(
        self,
        actual: np.ndarray | list[float],
        previous: np.ndarray | list[float],
        previous_previous: np.ndarray | list[float] | None = None,
    ) -> tuple[QuantizedTransportPacket, PredictiveTransportStats]:
        actual_array = as_float_array(actual, name="actual")
        predicted = predict_next_activation(
            previous,
            previous_previous,
            velocity_scale=self.velocity_scale,
        )
        if predicted.shape != actual_array.shape:
            raise ValueError(
                "Predicted and actual activations must share the same shape: "
                f"{predicted.shape} != {actual_array.shape}"
            )

        residual = actual_array - predicted
        max_abs = float(np.max(np.abs(residual))) if residual.size else 0.0
        qmax = (2 ** (self.bit_width - 1)) - 1
        scale = max_abs / qmax if max_abs else 1.0
        quantized = np.round(residual / scale).clip(-qmax, qmax).astype(_quantized_dtype(self.bit_width))

        packet = QuantizedTransportPacket(
            quantized_residual=quantized,
            scale=scale,
            bit_width=self.bit_width,
        )
        reconstructed = self.decode(packet, previous, previous_previous)

        raw_norm = float(np.linalg.norm(actual_array.ravel()))
        residual_norm = float(np.linalg.norm(residual.ravel()))
        compressed_bits = max(packet.payload_bytes * 8, 1)
        original_bits = max(actual_array.size * self.raw_bits_per_value, 1)

        stats = PredictiveTransportStats(
            raw_norm=raw_norm,
            residual_norm=residual_norm,
            residual_ratio=residual_norm / max(raw_norm, 1e-12),
            compression_ratio=float(original_bits / compressed_bits),
            reconstruction=reconstruction_stats(actual_array, reconstructed),
        )
        return packet, stats

    def decode(
        self,
        packet: QuantizedTransportPacket,
        previous: np.ndarray | list[float],
        previous_previous: np.ndarray | list[float] | None = None,
    ) -> np.ndarray:
        predicted = predict_next_activation(
            previous,
            previous_previous,
            velocity_scale=self.velocity_scale,
        )
        if predicted.shape != packet.quantized_residual.shape:
            raise ValueError(
                "Predicted activations and quantized residual must share the same shape: "
                f"{predicted.shape} != {packet.quantized_residual.shape}"
            )
        residual = packet.quantized_residual.astype(float) * packet.scale
        return predicted + residual


__all__ = [
    "PredictiveTransportCodec",
    "PredictiveTransportStats",
    "QuantizedTransportPacket",
    "predict_next_activation",
]
