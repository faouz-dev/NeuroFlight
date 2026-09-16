"""Benchmark a 10-population motor-neuron readout against the legacy 4-value readout.

This does not change the flight controller.  It asks a narrower question:
does preserving b1, b2, b3, i1 and i2 separately on each body side produce a
less ambiguous neural control signal than aggregating them into four values?
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.demo_neuroflight_3d_v2 import NeuroFlight3AxisControllerV2
from scripts.validate_neural_control_loop import AXES, PROBE_RATE_RAD_S, SEEDS, latency_ms, sample_step

OUT = Path("logs/motor10_benchmark") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)


class Motor10Readout(NeuroFlight3AxisControllerV2):
    """Same sensory encoder and LIF model, but retain ten motor populations."""

    def __init__(self, *args, **kwargs):
        self._defer_calibration = True
        super().__init__(*args, **kwargs)
        self.traces = np.zeros(len(self.motor_masks), dtype=float)
        self._defer_calibration = False
        self.calibrate_decoder()

    def _build_motor_masks(self):
        names = []
        masks = []
        for motor in ("b1 MN", "b2 MN", "b3 MN", "i1 MN", "i2 MN"):
            for side in ("L", "R"):
                names.append(f"{motor.replace(' ', '_')}_{side}")
                masks.append(self._make_mask(self._by_types([motor], side)))
        self.motor_names = tuple(names)
        self.motor_masks = tuple(masks)

    def calibrate_decoder(self):
        if not self._defer_calibration:
            super().calibrate_decoder()

    def _feature_vector(self):
        return self.traces.copy()

    def _tick(self, rng, command_xyz):
        rates = self._sensory_rates(command_xyz)
        probability = np.clip(rates * self.brain.dt_ms / 1000.0, 0.0, 1.0)
        forced = self.sensory_indices[rng.random(len(self.sensory_indices)) < probability]
        spikes, _ = self.brain.step(forced_spikes=forced)
        counts = np.asarray([mask[spikes].sum() for mask in self.motor_masks], dtype=float)
        self.traces = self.traces * self.trace_decay + counts
        return self._feature_vector()


def main() -> None:
    controller = Motor10Readout(seed=4242, verbose=True)
    gains = np.zeros((len(SEEDS), 3, 3), dtype=float)
    delays = np.full((len(SEEDS), 3), np.nan, dtype=float)

    for seed_index, seed in enumerate(SEEDS):
        for axis in range(3):
            command = np.zeros(3)
            command[axis] = 1.0
            times, response = sample_step(controller, command, seed + axis * 100)
            gains[seed_index, :, axis] = response[int(.7 * len(response)):].mean(axis=0)
            delays[seed_index, axis] = latency_ms(times, response[:, axis])

    mean = gains.mean(axis=0)
    std = gains.std(axis=0)
    diagonal = np.diag(mean)
    cross = mean.copy()
    np.fill_diagonal(cross, 0.0)
    report = {
        "motor_populations": controller.motor_names,
        "mean_step_gain": mean.tolist(),
        "step_gain_std": std.tolist(),
        "median_latency_ms": np.nanmedian(delays, axis=0).tolist(),
        "cross_talk_ratio": float(np.max(np.abs(cross)) / max(float(np.min(np.abs(diagonal))), 1e-9)),
        "seed_cv": float(np.max(np.diag(std) / np.maximum(np.abs(diagonal), 1e-9))),
        "note": "Compare this report with neural_loop_validation before changing flight control.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOTOR-10 READOUT BENCHMARK")
    print("Mean gain (rows=decoded, columns=input):")
    print(np.array2string(mean, precision=4))
    print("Median latency ms:", np.array2string(np.nanmedian(delays, axis=0), precision=1))
    print(f"Cross-talk ratio: {report['cross_talk_ratio']:.3f}; seed CV: {report['seed_cv']:.3f}")
    print("Files:", OUT)


if __name__ == "__main__":
    main()
