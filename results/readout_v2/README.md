# Readout comparison: independent stochastic episodes

Status: experimental, NOT qualified for closed-loop flight.

## Protocol

MaleCNS v1.0 cache: 165,122 traced neurons, 6,235,682 connections,
minimum synapse weight 5. Existing LIF dynamics and sensory encoder unchanged.
Ten individual motor traces (b1, b2, b3, i1, i2 on each side), dt 0.5 ms,
trace time constant 100 ms. Targets are effective angular rates in rad/s;
they are not biological forces or drone actuator commands.

Each independent episode starts from a reset, receives 600 ms of neutral
stimulation and 600 ms of constant stimulation. The feature is the mean of
20 trace samples in the final 200 ms. The optional baseline uses only the
preceding neutral interval, never a future counterfactual.

39 training episodes, 13 validation episodes, and 18 final test episodes.
All seeds are disjoint. Training/validation use isolated +/-0.7 rad/s and
mixed +/-0.4 rad/s inputs. Final tests use isolated +/-0.4 rad/s and two new
mixed vectors, with two independent seeds per target.

Four feature representations times four ridge penalties = 16 candidates.
Selection uses validation only. The selected candidate is motor10, penalty
0.1; its training-fitted normalization and coefficients are in report.json.
The legacy3 comparison is refitted on the same training data and selected
on validation, so it is NOT the original old flight decoder.

## Final test results

RMSE in rad/s, lower is better:

| Readout | Roll | Pitch | Yaw | Mean of axis RMSE |
|---|---:|---:|---:|---:|
| Selected motor10 | 0.16844 | 0.28659 | 0.21089 | 0.22197 |
| Refit legacy3 | 0.14331 | 0.29252 | 0.22027 | 0.21870 |
| Always zero | 0.23570 | 0.30185 | 0.21082 | 0.24946 |

The selected candidate improves mean error by about 11% against always zero,
but is about 1.5% worse than the refitted legacy representation. The yaw score
is effectively tied with always zero. Pitch improvement is small. These small
samples do not establish statistical significance, response speed, closed-loop
stability, or an optimal decoder. No candidate is promoted to flight control.

The encoder inspection finds roll/yaw cosine overlap 0.84968, despite full
column rank. This is a measured overlap of engineered input patterns, not a
biological claim or proof of the cause of decoding errors. A controlled future
comparison of sensory encodings would require a fresh final test set.

## Reproduction and interruption recovery

Run from the repository root:

```sh
python -m scripts.download_malecns_data --build-cache
python -m unittest scripts.test_readout_v2 -v
python -m scripts.compare_readout_v2
python -m scripts.inspect_encoder_overlap
```

The three checked-in episode files contain all observations used for fitting
and evaluation, including the exact seeds and targets. Completed episodes are
skipped on restart. A restart with these files reproduced the identical report.
To run a genuinely new experiment, use a separate output directory and change
the protocol/seeds explicitly; do not mix changed experiments in one checkpoint.

Runtime: numpy 2.3.5, pandas 2.2.3, pyarrow 25.0.1, mujoco 3.13.0,
mujoco-menagerie 2026.9.0. The comparison imports the existing simulator module
but runs neural experiments only, not vehicle dynamics.

Five numerical tests passed (aggregation, centering, linear recovery, constant
features, zero reference). Raw trace time series are not retained in these
compact checkpoints; regenerate them from the specified seeds if needed.
