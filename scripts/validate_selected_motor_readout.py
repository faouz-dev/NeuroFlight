"""Run dynamic neural-loop validation for the held-out selected motor readout."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.benchmark_motor10_readout import Motor10Readout
from scripts.validate_neural_control_loop import AXES, PROBE_RATE_RAD_S, SEEDS, latency_ms, sample_step

# Best held-out selection reported by scripts.search_motor_readout.
SELECTED = (0, 1, 3, 4, 5, 7, 9)
OUT = Path("logs/selected_motor_readout") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)


class SelectedMotorReadout(Motor10Readout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected = np.asarray(SELECTED, dtype=int)
        self.calibrate_decoder()

    def _feature_vector(self):
        if not hasattr(self, "selected"):
            return super()._feature_vector()
        return self.traces[self.selected].copy()


def main() -> None:
    controller = SelectedMotorReadout(seed=4242, verbose=True)
    gains = np.zeros((len(SEEDS), 3, 3), dtype=float)
    delays = np.full((len(SEEDS), 3), np.nan, dtype=float)
    for seed_index, seed in enumerate(SEEDS):
        for axis in range(3):
            command = np.zeros(3)
            command[axis] = 1.0
            times, response = sample_step(controller, command, seed + 100 * axis)
            gains[seed_index, :, axis] = response[int(.7 * len(response)):].mean(axis=0)
            delays[seed_index, axis] = latency_ms(times, response[:, axis])

    mean, std = gains.mean(axis=0), gains.std(axis=0)
    diagonal = np.diag(mean)
    cross = mean.copy()
    np.fill_diagonal(cross, 0.0)
    report = {
        "selected_populations": [controller.motor_names[index] for index in SELECTED],
        "mean_step_gain": mean.tolist(),
        "step_gain_std": std.tolist(),
        "median_latency_ms": np.nanmedian(delays, axis=0).tolist(),
        "cross_talk_ratio": float(np.max(np.abs(cross)) / max(float(np.min(np.abs(diagonal))), 1e-9)),
        "seed_cv": float(np.max(np.diag(std) / np.maximum(np.abs(diagonal), 1e-9))),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("SELECTED MOTOR READOUT VALIDATION")
    print("Populations:", ", ".join(report["selected_populations"]))
    print(np.array2string(mean, precision=4))
    print("Median latency ms:", np.array2string(np.nanmedian(delays, axis=0), precision=1))
    print(f"Cross-talk ratio: {report['cross_talk_ratio']:.3f}; seed CV: {report['seed_cv']:.3f}")
    print("Files:", OUT)


if __name__ == "__main__":
    main()
