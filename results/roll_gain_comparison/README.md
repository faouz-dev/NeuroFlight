# Reduced attitude feedback: paired comparison

The decoder, damping (3/s), output filter (50 ms), noise seed and initial state are paired. Only the outer attitude coefficient changes: 2/s to 1/s. This modifies the engineered sensory input, not the connectome.

Diagnostic seed 1100001 was previously observed; confirmation seed 1200001 is new. Each trial lasts 4 s. Both initial signs are tested at 12 degrees and 0.15 rad/s. No direct attitude or gyro bypass supplies the neural roll torque. Altitude hold is classical.

| Seed | Initial roll | Variant | Last 0.5 s mean absolute roll | Final roll | Final rate (rad/s) | Pass |
|---|---:|---|---:|---:|---:|---|
| 1100001 | +12 | baseline | 2.714 | 3.488 | 0.0102 | True |
| 1100001 | +12 | reduced | 2.228 | -0.541 | 0.0903 | True |
| 1100001 | -12 | baseline | 10.450 | 8.134 | -0.2008 | False |

Completed trials: 3/8.

Pass requires a full run within bounds, mean absolute roll below 5 degrees over the last 0.5 s, and final absolute rate below 0.2 rad/s. It does not require final absolute roll below 5 degrees. Short runs and two seeds cannot establish reliable stabilization. Do not infer biological validation, 3-axis control, or hardware readiness.

Reproduce:

```bash
python -m scripts.download_malecns_data --build-cache
OPENBLAS_NUM_THREADS=1 python -m scripts.roll_gain_comparison
OPENBLAS_NUM_THREADS=1 python -m scripts.roll_gain_comparison --confirmation
python -m scripts.report_roll_gain_comparison
```
