"""Frozen-decoder paired roll trials; atomic, fingerprinted resume checkpoints."""
import hashlib
import json
from pathlib import Path
from scripts import roll_closed_loop as roll

OUT = Path('results/roll_paired_confirmation')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    decoder = json.loads(Path('results/roll_closed_loop_pilot_robust/decoder.json').read_text())
    protocol = dict(decoder=decoder, attitude_k=roll.ATTITUDE_K,
                    damping=roll.DAMPING, output_tau=roll.OUTPUT_TAU,
                    duration=roll.DURATION, scenarios=roll.SCENARIOS,
                    modes=['oracle', 'none', 'neural', 'disconnected'],
                    signs=[1, -1])
    sources = ['scripts/roll_closed_loop.py', 'scripts/roll_paired_confirmation.py',
               'scripts/demo_neuroflight_3d_v2.py', 'scripts/compare_readout_v2.py',
               'scripts/benchmark_motor10_readout.py', 'flybrain/lif.py']
    protocol['source_sha256'] = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    manifest = OUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != protocol:
        # JSON turns tuples into lists, so compare canonical serializations.
        if json.dumps(json.loads(manifest.read_text()), sort_keys=True) != json.dumps(protocol, sort_keys=True):
            raise RuntimeError('Protocol changed: use a new output directory')
    manifest.write_text(json.dumps(protocol, indent=2))
    brain = roll.RawMotors(verbose=False)
    vehicle = roll.build_vehicle()
    summaries = []
    for angle, rate, seed in roll.SCENARIOS:
        for sign in (1, -1):
            for mode in protocol['modes']:
                path = OUT/f'{seed}_{sign:+d}_{mode}.json'
                if path.exists():
                    result = json.loads(path.read_text())
                    if result.get('protocol_sha256') != fingerprint:
                        raise RuntimeError(f'Incompatible checkpoint: {path}')
                else:
                    result = roll.experiment(brain, decoder, vehicle, mode, sign*angle, sign*rate, seed)
                    result['protocol_sha256'] = fingerprint
                    tmp = path.with_suffix('.tmp')
                    tmp.write_text(json.dumps(result, allow_nan=False))
                    tmp.replace(path)
                summary = {k: v for k, v in result.items() if k != 'trace'}
                summaries.append(summary)
                tmp = OUT/'summary.tmp'
                tmp.write_text(json.dumps(summaries, indent=2, allow_nan=False))
                tmp.replace(OUT/'summary.json')
                print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
