"""Report all completed trials, preserving failed candidates and raw traces."""
import json
from pathlib import Path

OUT = Path('results/roll_gain_comparison')


def main():
    rows = []
    for path in sorted(OUT.glob('1*.json')):
        result = json.loads(path.read_text())
        rows.append({k:v for k,v in result.items() if k != 'trace'})
    (OUT/'summary.json').write_text(json.dumps(rows, indent=2, allow_nan=False))
    lines = ['# Reduced attitude feedback: paired comparison', '',
             'The decoder, damping (3/s), output filter (50 ms), noise seed and initial state are paired. '
             'Only the outer attitude coefficient changes: 2/s to 1/s. '
             'This modifies the engineered sensory input, not the connectome.', '',
             'Diagnostic seed 1100001 was previously observed; confirmation seed 1200001 is new. '
             'Each trial lasts 4 s. Both initial signs are tested at 12 degrees and 0.15 rad/s. '
             'No direct attitude or gyro bypass supplies the neural roll torque. Altitude hold is classical.', '',
             '| Seed | Initial roll | Variant | Last 0.5 s mean absolute roll | Final roll | Final rate (rad/s) | Pass |',
             '|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        tail = r['tail_mean_abs_roll_deg']
        tail_s = f'{tail:.3f}' if tail is not None else 'bounds failure'
        lines.append(f"| {r['seed']} | {r['initial_roll_deg']:+g} | {r['variant']} | {tail_s} | {r['final_roll_deg']:.3f} | {r['final_rate']:.4f} | {r['passed']} |")
    lines += ['', f'Completed trials: {len(rows)}/8.', '',
              'Pass requires a full run within bounds, mean absolute roll below 5 degrees over the last 0.5 s, '
              'and final absolute rate below 0.2 rad/s. It does not require final absolute roll below 5 degrees. '
              'Short runs and two seeds cannot establish reliable stabilization. '
              'Do not infer biological validation, 3-axis control, or hardware readiness.', '',
              'Reproduce:', '', '```bash',
              'python -m scripts.download_malecns_data --build-cache',
              'OPENBLAS_NUM_THREADS=1 python -m scripts.roll_gain_comparison',
              'OPENBLAS_NUM_THREADS=1 python -m scripts.roll_gain_comparison --confirmation',
              'python -m scripts.report_roll_gain_comparison', '```', '']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
