"""Open-loop diagnostic, NOT a flight qualification.

For each fresh seed, replay neutral and signed perturbations with identical
initial state and random numbers. This subtracts ongoing neutral drift rather
than confusing it with the response. No muscle-force interpretation is made.
"""
import json
from pathlib import Path
import numpy as np
from scripts.benchmark_motor10_readout import Motor10Readout

SEEDS = (820001, 820002, 820003, 820004)
AMPLITUDE = 0.7  # effective rate in rad/s, NOT normalized encoder command


def trial(controller, rate, seed):
    controller.reset_runtime(seed=seed)
    dt = controller.brain.dt_ms
    for _ in range(round(600 / dt)):
        controller.step_from_effective_rate(np.zeros(3), 1)
    samples = []
    raw = []
    for step in range(round(700 / dt)):
        out = controller.step_from_effective_rate(rate, 1)
        if step >= round(500 / dt):
            samples.append(out['decoded'].copy())
            raw.append(controller.traces.copy())
    return np.mean(samples, axis=0), np.mean(raw, axis=0)


def main():
    controller = Motor10Readout(seed=4242, verbose=False)
    matrices, raw_matrices, records = [], [], []
    for seed in SEEDS:
        neutral, raw_neutral = trial(controller, np.zeros(3), seed)
        matrix, raw_matrix = np.zeros((3, 3)), np.zeros((10, 3))
        for axis in range(3):
            rate = np.eye(3)[axis] * AMPLITUDE
            positive, raw_positive = trial(controller, rate, seed)
            negative, raw_negative = trial(controller, -rate, seed)
            matrix[:, axis] = (positive-negative)/(2*AMPLITUDE)
            raw_matrix[:, axis] = (raw_positive-raw_negative)/(2*AMPLITUDE)
            records.append(dict(seed=seed, axis=axis, neutral=neutral.tolist(),
                                positive=positive.tolist(), negative=negative.tolist(),
                                positive_effect=(positive-neutral).tolist(),
                                negative_effect=(negative-neutral).tolist(),
                                raw_neutral=raw_neutral.tolist()))
        matrices.append(matrix)
        raw_matrices.append(raw_matrix)
        print('Finished seed', seed, flush=True)
    matrices = np.asarray(matrices)
    raw_matrices = np.asarray(raw_matrices)
    result = dict(protocol='paired +/-/neutral, same seed; open-loop only',
                  amplitude_rad_s=AMPLITUDE, seeds=SEEDS,
                  motor_names=controller.motor_names,
                  mean_decoded_slope=matrices.mean(0).tolist(),
                  std_decoded_slope=matrices.std(0, ddof=1).tolist(),
                  decoded_slopes_by_seed=matrices.tolist(),
                  mean_raw_slope=raw_matrices.mean(0).tolist(),
                  raw_slopes_by_seed=raw_matrices.tolist(), records=records,
                  limitations='Four seeds, single amplitude, settled mean only. '
                  'No independent flight validation, no latency measurement.')
    path = Path('logs/paired_neural_probe/report.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('Mean decoded slopes per rad/s:\n', matrices.mean(0))
    print('Sample standard deviation:\n', matrices.std(0, ddof=1))
    print(path)


if __name__ == '__main__':
    main()
