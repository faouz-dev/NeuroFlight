"""Seed-separated, open-loop readout selection. Never authorizes flight.

Targets are effective angular rates in rad/s, not torques. Only observed motor
traces enter the decoder. No target, future neutral or sensor shortcut is used.
Train/validation/test contain disjoint random seeds; test includes new amplitudes.
Episode means measure settled information, not real-time control performance.
"""
import json
from pathlib import Path
import numpy as np
from scripts.benchmark_motor10_readout import Motor10Readout
from scripts.demo_neuroflight_3d_v2 import GYRO_TO_HZ, DELTA_RATE_HZ

OUT = Path('results/readout_v2')
KINDS = ('legacy3', 'motor10', 'centered10', 'raw_and_centered20')
LAMBDAS = (0.1, 1., 10., 100.)


class RawMotors(Motor10Readout):
    def calibrate_decoder(self):
        pass  # Collect once, fit candidate readouts offline.


def features(raw, neutral, kind):
    if kind == 'motor10':
        return raw
    if kind == 'centered10':
        return raw-neutral
    if kind == 'raw_and_centered20':
        return np.concatenate((raw, raw-neutral), axis=-1)
    if kind == 'legacy3':
        b12l, b12r = raw[..., 0]+raw[..., 2], raw[..., 1]+raw[..., 3]
        b3il, b3ir = raw[..., 4]+raw[..., 6], raw[..., 5]+raw[..., 7]
        return np.stack((b12l-b12r, b12l+b12r-b3il-b3ir, b3il-b3ir), axis=-1)
    raise ValueError(kind)


def collect(brain, target, seed):
    brain._reset_brain_state()
    rng = np.random.default_rng(seed)
    dt = brain.brain.dt_ms
    baseline = []
    for i in range(round(600/dt)):
        raw = brain._tick(rng, np.zeros(3))
        if i >= round(400/dt):
            baseline.append(raw.copy())
    neutral = np.mean(baseline, axis=0)
    command = np.clip(np.asarray(target)*GYRO_TO_HZ/DELTA_RATE_HZ, -1, 1)
    samples = []
    for i in range(round(600/dt)):
        raw = brain._tick(rng, command)
        if i % round(10/dt) == 0:
            samples.append(raw.copy())
    samples = np.asarray(samples)
    return dict(seed=seed, target=list(target), neutral=neutral.tolist(),
                raw=samples[-20:].mean(axis=0).tolist(), trace=samples.tolist())


def arrays(rows, kind):
    raw = np.array([r['raw'] for r in rows])
    neutral = np.array([r['neutral'] for r in rows])
    y = np.array([r['target'] for r in rows])
    return features(raw, neutral, kind), y


def fit(x, y, penalty):
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    a = np.column_stack((np.ones(len(x)), (x-mean)/scale))
    reg = np.eye(a.shape[1])*penalty
    reg[0, 0] = 0
    w = np.linalg.solve(a.T@a+reg, a.T@y)
    return dict(mean=mean.tolist(), scale=scale.tolist(), weights=w.tolist())


def predict(model, x):
    a = np.column_stack((np.ones(len(x)), (x-model['mean'])/model['scale']))
    return a@np.asarray(model['weights'])


def metrics(y, pred):
    rmse = np.sqrt(np.mean((pred-y)**2, axis=0))
    return dict(rmse_rad_s=rmse.tolist(), mean_rmse=float(rmse.mean()))


def evaluate(train, validation, test):
    candidates = []
    for kind in KINDS:
        x, y = arrays(train, kind)
        xv, yv = arrays(validation, kind)
        for penalty in LAMBDAS:
            model = fit(x, y, penalty)
            score = metrics(yv, predict(model, xv))
            candidates.append(dict(kind=kind, penalty=penalty, model=model, validation=score))
    candidates.sort(key=lambda c: c['validation']['mean_rmse'])
    best = candidates[0]  # Freeze choice BEFORE looking at final test.
    xt, yt = arrays(test, best['kind'])
    test_prediction = predict(best['model'], xt)
    result = dict(best=best, validation_candidates=[{k:v for k,v in c.items() if k!='model'} for c in candidates],
                  test=metrics(yt, test_prediction), zero_baseline=metrics(yt, np.zeros_like(yt)),
                  test_targets=yt.tolist(), test_predictions=test_prediction.tolist(),
                  train_episodes=len(train), validation_episodes=len(validation), test_episodes=len(test),
                  flight_validated=False,
                  limitations='Small static experiment. Test values are 200 ms means; '
                  'performance does not establish feedback stability or biological force meaning.')
    # Compare best legacy feature model chosen ONLY on validation data.
    legacy = next(c for c in candidates if c['kind']=='legacy3')
    xl, _ = arrays(test, 'legacy3')
    result['legacy_refit_test'] = metrics(yt, predict(legacy['model'], xl))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    known = [np.zeros(3)] + [s*.7*np.eye(3)[a] for a in range(3) for s in (-1,1)]
    known += [np.array(c)*.4 for c in ((1,1,0),(-1,-1,0),(1,0,1),(-1,0,-1),(0,1,-1),(0,-1,1))]
    unseen = [np.zeros(3)] + [s*.4*np.eye(3)[a] for a in range(3) for s in (-1,1)]
    unseen += [np.array([.3,-.5,.2]), np.array([-.3,.5,-.2])]
    splits = {'train': (known, (910000,920000,930000)),
              'validation': (known, (940000,)), 'test': (unseen, (950000,960000))}
    brain = RawMotors(verbose=False)
    rows = {}
    for split, (commands, seeds) in splits.items():
        rows[split] = []
        for seed in seeds:
            for i, command in enumerate(commands):
                rows[split].append(collect(brain, command, seed+i))
            print(split, 'seed', seed, 'complete', flush=True)
            (OUT / (split+'.json')).write_text(json.dumps(rows[split]))
    result = evaluate(**rows)
    result['motor_names'] = brain.motor_names
    (OUT/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k:result[k] for k in ('test','zero_baseline','legacy_refit_test')}, indent=2))
    print('Selected:', result['best']['kind'], result['best']['penalty'])


if __name__ == '__main__':
    main()
