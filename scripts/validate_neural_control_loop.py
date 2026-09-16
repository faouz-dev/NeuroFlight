"""Measure whether the neural readout is safe to close around attitude control.

This is deliberately upstream of MuJoCo flight.  It checks the properties a
static decoder calibration cannot see: sign, cross-axis leakage, latency and
seed-to-seed variation.  Do not use a flight demo when this script reports
FAIL.

Run from the repository root:

    python -m scripts.validate_neural_control_loop
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.demo_neuroflight_3d_v2 import NeuroFlight3AxisControllerV2

AXES = ("roll", "pitch", "yaw")
PROBE_RATE_RAD_S = 0.7
WARMUP_MS = 350.0
STEP_MS = 700.0
SAMPLE_EVERY_MS = 5.0
SEEDS = (31_001, 31_002, 31_003, 31_004)

# Engineering gates, not claims about fly biology.
MAX_LATENCY_MS = 180.0
MIN_DIAGONAL_GAIN = 0.10
MAX_CROSSTALK_RATIO = 0.75
MAX_SEED_CV = 0.70

OUT = Path("logs/neural_loop_validation") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)


def sample_step(controller, command: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return time series of decoded output centered on this session's neutral."""
    controller.reset_runtime(seed=seed)
    dt = controller.brain.dt_ms
    warm_steps = int(round(WARMUP_MS / dt))
    step_steps = int(round(STEP_MS / dt))
    stride = max(1, int(round(SAMPLE_EVERY_MS / dt)))

    neutral_samples = []
    for index in range(warm_steps):
        value = controller.step_from_effective_rate(np.zeros(3), 1)["decoded"]
        if index % stride == 0:
            neutral_samples.append(value)
    neutral = np.mean(np.asarray(neutral_samples), axis=0)

    values = []
    times = []
    effective_rate = PROBE_RATE_RAD_S * np.asarray(command, dtype=float)
    for index in range(step_steps):
        value = controller.step_from_effective_rate(effective_rate, 1)["decoded"] - neutral
        if index % stride == 0:
            values.append(value)
            times.append((index + 1) * dt)
    return np.asarray(times), np.asarray(values)


def latency_ms(times: np.ndarray, signal: np.ndarray) -> float:
    """First sustained 20%-of-final crossing; NaN means no usable response."""
    tail = signal[max(1, int(0.7 * len(signal))):]
    final = float(np.mean(tail))
    if abs(final) < 1e-9:
        return float("nan")
    threshold = 0.20 * abs(final)
    # A single 0.5 ms sample can cross the threshold because the LIF output is
    # stochastic.  Require five sampled points (25 ms at the default stride)
    # in the final response direction before reporting a control-relevant delay.
    directed = signal * np.sign(final)
    sustained = np.convolve(directed >= threshold, np.ones(5, dtype=int), mode="valid") == 5
    crossed = np.flatnonzero(sustained)
    return float(times[crossed[0] + 4]) if len(crossed) else float("nan")


def main() -> None:
    controller = NeuroFlight3AxisControllerV2(seed=4242, verbose=True)
    rows: list[dict] = []
    gains = np.zeros((len(SEEDS), 3, 3), dtype=float)
    delays = np.full((len(SEEDS), 3), np.nan, dtype=float)

    for seed_index, seed in enumerate(SEEDS):
        for axis, name in enumerate(AXES):
            command = np.zeros(3)
            command[axis] = 1.0
            times, response = sample_step(controller, command, seed + axis * 100)
            tail = response[max(1, int(0.7 * len(response))):]
            gain = np.mean(tail, axis=0)
            gains[seed_index, :, axis] = gain
            delays[seed_index, axis] = latency_ms(times, response[:, axis])

            for time_ms, value in zip(times, response):
                rows.append({
                    "seed": seed,
                    "input_axis": name,
                    "time_ms": float(time_ms),
                    "decoded_roll_centered": float(value[0]),
                    "decoded_pitch_centered": float(value[1]),
                    "decoded_yaw_centered": float(value[2]),
                })

    mean_gain = gains.mean(axis=0)
    gain_std = gains.std(axis=0)
    diagonal = np.diag(mean_gain)
    diagonal_std = np.diag(gain_std)
    cross = mean_gain.copy()
    np.fill_diagonal(cross, 0.0)
    cross_ratio = float(np.max(np.abs(cross)) / max(float(np.min(np.abs(diagonal))), 1e-9))
    seed_cv = float(np.max(diagonal_std / np.maximum(np.abs(diagonal), 1e-9)))
    median_delay = np.nanmedian(delays, axis=0)
    sign_ok = bool(np.all(diagonal > MIN_DIAGONAL_GAIN))
    delay_ok = bool(np.all(np.isfinite(median_delay)) and np.all(median_delay <= MAX_LATENCY_MS))
    passed = sign_ok and delay_ok and cross_ratio <= MAX_CROSSTALK_RATIO and seed_cv <= MAX_SEED_CV

    with (OUT / "step_responses.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "pass": bool(passed),
        "axes": AXES,
        "probe_rate_rad_s": PROBE_RATE_RAD_S,
        "mean_step_gain": mean_gain.tolist(),
        "step_gain_std": gain_std.tolist(),
        "median_latency_ms": median_delay.tolist(),
        "cross_talk_ratio": cross_ratio,
        "seed_coefficient_of_variation": seed_cv,
        "criteria": {
            "minimum_diagonal_gain": MIN_DIAGONAL_GAIN,
            "maximum_latency_ms": MAX_LATENCY_MS,
            "maximum_cross_talk_ratio": MAX_CROSSTALK_RATIO,
            "maximum_seed_cv": MAX_SEED_CV,
        },
        "interpretation": (
            "PASS means only that the current neural readout is dynamically "
            "suitable for a guarded MuJoCo closed-loop test. It does not "
            "validate hardware flight or establish a biological muscle-to-force map."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("NEUROFLIGHT DYNAMIC NEURAL-LOOP VALIDATION")
    print("PASS:", passed)
    print("Mean gain (rows=decoded, columns=input):")
    print(np.array2string(mean_gain, precision=4))
    print("Median latency ms:", np.array2string(median_delay, precision=1))
    print(f"Cross-talk ratio: {cross_ratio:.3f}; seed CV: {seed_cv:.3f}")
    print("Files:", OUT)


if __name__ == "__main__":
    main()
