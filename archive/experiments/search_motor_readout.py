"""Select a compact, reproducible readout from ten motor-neuron populations.

The selector never evaluates on the random seed used to fit a readout.  It
therefore identifies combinations that survive new stochastic spike trains,
before they are considered for the dynamic closed-loop benchmark.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.benchmark_motor10_readout import Motor10Readout

RIDGE = 1.0
OUT = Path("logs/motor_readout_search") / datetime.now().strftime("%Y%m%d_%H%M%S")
OUT.mkdir(parents=True, exist_ok=True)


def collect(controller, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []
    for point_index, command in enumerate(controller._training_points()):
        samples = controller._collect_training_samples(command, seed=seed_base + point_index)
        features.append(samples)
        targets.append(np.repeat(command[None, :], len(samples), axis=0))
    return np.concatenate(features), np.concatenate(targets)


def fit_predict(x_train, y_train, x_test) -> np.ndarray:
    mean = x_train.mean(axis=0)
    std = np.maximum(x_train.std(axis=0), 1e-6)
    z_train = (x_train - mean) / std
    z_test = (x_test - mean) / std
    design = np.column_stack([np.ones(len(z_train)), z_train])
    test_design = np.column_stack([np.ones(len(z_test)), z_test])
    penalty = np.eye(design.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return test_design @ weights


def main() -> None:
    controller = Motor10Readout(seed=4242, verbose=True)
    x_train, y_train = collect(controller, 1_400_000)
    x_test, y_test = collect(controller, 1_500_000)
    results = []

    for size in range(1, len(controller.motor_names) + 1):
        for chosen in itertools.combinations(range(len(controller.motor_names)), size):
            prediction = fit_predict(x_train[:, chosen], y_train, x_test[:, chosen])
            axis_rmse = np.sqrt(np.mean((prediction - y_test) ** 2, axis=0))
            results.append({
                "indices": list(chosen),
                "populations": [controller.motor_names[index] for index in chosen],
                "n_populations": size,
                "rmse_per_axis": axis_rmse.tolist(),
                "mean_rmse": float(axis_rmse.mean()),
                # Fewer populations win only when their error is effectively tied.
                "score": float(axis_rmse.mean() + 0.002 * size),
            })

    results.sort(key=lambda item: item["score"])
    report = {
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "ridge_lambda": RIDGE,
        "all_populations": controller.motor_names,
        "best": results[0],
        "top_20": results[:20],
        "interpretation": (
            "This is a held-out neural decoding score, not permission to fly. "
            "The selected readout must still pass the dynamic loop validation."
        ),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("MOTOR READOUT SEARCH")
    print("Train/test samples:", len(x_train), "/", len(x_test))
    print("Best populations:", ", ".join(report["best"]["populations"]))
    print("RMSE roll/pitch/yaw:", np.array2string(np.asarray(report["best"]["rmse_per_axis"]), precision=4))
    print("Files:", OUT)


if __name__ == "__main__":
    main()
