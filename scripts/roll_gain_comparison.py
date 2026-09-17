"""Bounded paired comparison: reduce attitude feedback, keep neural decoder frozen.

Diagnostic seed was previously observed. Confirmation seed is new. This is a
four-second isolated-roll experiment, not a validated flight controller.
"""
import hashlib
import json
from pathlib import Path
import argparse
from scripts import roll_closed_loop as roll

OUT = Path('results/roll_gain_comparison')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    decoder = json.loads(Path('results/roll_closed_loop_pilot_robust/decoder.json').read_text())
    protocol = dict(decoder=decoder, gains={'baseline': 2.0, 'reduced': 1.0},
                    damping=roll.DAMPING, output_tau=roll.OUTPUT_TAU,
                    duration=roll.DURATION, angles=[12.0, -12.0],
                    diagnostic_seed=1100001, confirmation_seed=1200001,
                    source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in ['scripts/roll_closed_loop.py', 'scripts/roll_gain_comparison.py',
                                  'scripts/compare_readout_v2.py', 'flybrain/lif.py',
                                  'scripts/demo_neuroflight_3d_v2.py',
                                  'scripts/benchmark_motor10_readout.py']})
    serialized = json.dumps(protocol, sort_keys=True)
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()
    path = OUT/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise RuntimeError('Protocol changed; use a separate output directory')
    path.write_text(json.dumps(protocol, indent=2))
    brain = roll.RawMotors(verbose=False)
    vehicle = roll.build_vehicle()
    seed = protocol['confirmation_seed'] if args.confirmation else protocol['diagnostic_seed']
    for angle in protocol['angles']:
        for name, gain in protocol['gains'].items():
            roll.ATTITUDE_K = gain
            path = OUT/f'{seed}_{angle:+g}_{name}.json'
            if path.exists():
                result = json.loads(path.read_text())
                if result['protocol_sha256'] != fingerprint:
                    raise RuntimeError('Checkpoint fingerprint mismatch')
            else:
                result = roll.experiment(brain, decoder, vehicle, 'neural', angle, .15*(1 if angle>0 else -1), seed)
                result.update(variant=name, attitude_k=gain, protocol_sha256=fingerprint)
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(result, allow_nan=False))
                tmp.replace(path)
            print(json.dumps({k:v for k,v in result.items() if k!='trace'}), flush=True)


if __name__ == '__main__':
    main()
