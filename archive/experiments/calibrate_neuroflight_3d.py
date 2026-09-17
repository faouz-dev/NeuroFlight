"""Calibrate and validate the 3-axis MaleCNS -> Crazyflie adapter.

Run from the NeuroFlight repository root after copying this file to scripts/:

    python -m scripts.calibrate_neuroflight_3d

This is deliberately a *bench test*, not a flight attempt.  It measures the
neutral neural output, the complete 3 x 3 input/output mixing matrix, and the
error made when roll, pitch and yaw are requested simultaneously.  A later
flight controller should use calibration_3d.json only when this script reports
PASS.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.demo_neuroflight_3d_v2 import NeuroFlight3AxisControllerV2


# Effective angular-rate input to the V2 sensory adapter, in rad/s.
PROBE_RATE = 1.0
TRIALS = 3
WARM_MEASURE_MS = 500.0
SAMPLE_EVERY_MS = 5.0

# The exact limits are engineering acceptance criteria, not biological claims.
MAX_CONDITION_NUMBER = 12.0
MAX_COMBINED_RMSE = 0.20
MIN_AXIS_RESPONSE = 0.08

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/calibration_3d") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUT_DIR / "responses.csv"
JSON_PATH = OUT_DIR / "calibration_3d.json"
SUMMARY_PATH = OUT_DIR / "SUMMARY.txt"

AXES = ("roll", "pitch", "yaw")


def save_csv(path: Path, rows: list[dict]) -> None:
    # Single-axis and combined rows do not have exactly the same columns:
    # combined rows also contain pred_* and err_*.  Build the union before
    # handing rows to DictWriter instead of taking the first row as schema.
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def observe(controller, command: np.ndarray, seed: int) -> np.ndarray:
    """Return a time-average decoded vector after a fresh, neutral reset."""
    controller.reset_runtime(seed=seed)
    lif_steps = 1  # controller LIF dt and this calibration tick are both 0.5 ms
    steps = int(round(WARM_MEASURE_MS / controller.brain.dt_ms))
    stride = max(1, int(round(SAMPLE_EVERY_MS / controller.brain.dt_ms)))
    samples = []
    effective_rate = PROBE_RATE * np.asarray(command, dtype=float)

    for step in range(steps):
        decoded = controller.step_from_effective_rate(effective_rate, lif_steps)["decoded"]
        if step % stride == 0:
            samples.append(decoded.copy())

    return np.mean(np.asarray(samples), axis=0)


def mean_observation(controller, command: np.ndarray, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
    trials = np.asarray([
        observe(controller, command, seed_base + trial)
        for trial in range(TRIALS)
    ])
    return trials.mean(axis=0), trials.std(axis=0)


def main() -> None:
    print("=" * 76)
    print("NEUROFLIGHT — CALIBRATION 3D DU CANAL NEURONAL")
    print("=" * 76)
    controller = NeuroFlight3AxisControllerV2(seed=4242, verbose=True)

    rows: list[dict] = []
    neutral, neutral_std = mean_observation(controller, np.zeros(3), 10_000)
    rows.append({"kind": "neutral", "command": "0,0,0", "trial_mean": "all", **dict(zip(AXES, neutral)), **{f"std_{a}": v for a, v in zip(AXES, neutral_std)}})

    # Each matrix column is a central difference around the directly measured
    # neutral output.  It captures cross-talk instead of assuming it away.
    response = np.zeros((3, 3), dtype=float)
    individual_means = {}
    seed = 20_000

    for axis, name in enumerate(AXES):
        plus_command = np.zeros(3); plus_command[axis] = +1.0
        minus_command = np.zeros(3); minus_command[axis] = -1.0
        plus, plus_std = mean_observation(controller, plus_command, seed)
        minus, minus_std = mean_observation(controller, minus_command, seed + 100)
        seed += 1_000
        individual_means[(name, "+")] = plus
        individual_means[(name, "-")] = minus
        response[:, axis] = (plus - minus) / 2.0

        for sign, command, value, spread in (("+", plus_command, plus, plus_std), ("-", minus_command, minus, minus_std)):
            rows.append({"kind": f"single_{name}_{sign}", "command": ",".join(map(str, command.astype(int))), "trial_mean": "all", **dict(zip(AXES, value)), **{f"std_{a}": v for a, v in zip(AXES, spread)}})

    singular = np.linalg.svd(response, compute_uv=False)
    rank = int(np.linalg.matrix_rank(response, tol=1e-6))
    condition = float(singular[0] / max(singular[-1], 1e-12))
    inverse = np.linalg.pinv(response, rcond=1e-4)

    # Simultaneous points expose nonlinear saturation / shared-population
    # interference.  The prediction is the linear sum of independently
    # measured responses around the *measured* neutral point.
    combined_commands = [
        np.array([sx, sy, sz], dtype=float)
        for sx in (-0.60, +0.60)
        for sy in (-0.60, +0.60)
        for sz in (-0.60, +0.60)
    ]
    combined_errors = []
    for index, command in enumerate(combined_commands):
        actual, spread = mean_observation(controller, command, 50_000 + 100 * index)
        predicted = neutral + response @ command
        error = actual - predicted
        combined_errors.append(error)
        rows.append({
            "kind": "combined",
            "command": ",".join(f"{x:+.2f}" for x in command),
            "trial_mean": "all",
            **dict(zip(AXES, actual)),
            **{f"std_{a}": v for a, v in zip(AXES, spread)},
            **{f"pred_{a}": v for a, v in zip(AXES, predicted)},
            **{f"err_{a}": v for a, v in zip(AXES, error)},
        })

    combined_rmse = float(np.sqrt(np.mean(np.square(combined_errors))))
    axis_response = np.linalg.norm(response, axis=0)
    passed = bool(
        rank == 3
        and condition <= MAX_CONDITION_NUMBER
        and np.all(axis_response >= MIN_AXIS_RESPONSE)
        and combined_rmse <= MAX_COMBINED_RMSE
    )

    save_csv(CSV_PATH, rows)
    report = {
        "neutral_output": neutral.tolist(),
        "response_matrix_rows_decoded_roll_pitch_yaw_columns_input_roll_pitch_yaw": response.tolist(),
        "decoder_matrix_input_from_centered_output": inverse.tolist(),
        "singular_values": singular.tolist(),
        "rank": rank,
        "condition_number": condition,
        "axis_response_norm": axis_response.tolist(),
        "combined_rmse": combined_rmse,
        "acceptance": {
            "max_condition_number": MAX_CONDITION_NUMBER,
            "min_axis_response": MIN_AXIS_RESPONSE,
            "max_combined_rmse": MAX_COMBINED_RMSE,
        },
        "pass": passed,
        "note": "Use the neutral output directly; do not estimate neutral as (plus + minus) / 2.",
    }
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = [
        "NEUROFLIGHT 3D CALIBRATION",
        "",
        f"PASS: {passed}",
        f"Measured neutral output: {np.array2string(neutral, precision=4)}",
        f"Rank: {rank}/3",
        f"Condition number: {condition:.3f} (limit {MAX_CONDITION_NUMBER:.1f})",
        f"Axis response norms: {np.array2string(axis_response, precision=4)} (minimum {MIN_AXIS_RESPONSE:.2f})",
        f"Combined-input RMSE: {combined_rmse:.4f} (limit {MAX_COMBINED_RMSE:.2f})",
        "",
        "Response matrix: rows=decoded roll,pitch,yaw; columns=input roll,pitch,yaw",
        np.array2string(response, precision=5),
        "",
        "Interpretation:",
        "- PASS: a flight controller may load calibration_3d.json, then apply a separate safety envelope.",
        "- FAIL: do not run a flight demo. The selected haltere channels cannot yet provide a stable independent 3-axis adapter.",
    ]
    SUMMARY_PATH.write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\nFiles: {SUMMARY_PATH}\n       {JSON_PATH}\n       {CSV_PATH}")


if __name__ == "__main__":
    main()
