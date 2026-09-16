from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import math
import sys

import mujoco
import mujoco_menagerie as mm
import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


# ============================================================
# NEUROFLIGHT - FULL AXIS VALIDATION V4
#
# Corrects the V3 closed-loop decoder:
# V3 normalized (pos-neg)/(pos+neg), which saturated for pitch
# and compressed yaw. V4 calibrates the RAW motor differential
# automatically for every candidate:
#
#       diff = trace(pos motor) - trace(neg motor)
#
# and learns:
#       neutral center
#       positive scale
#       negative scale
#
# before running MuJoCo.
#
# One execution:
#   - screens roll/pitch/yaw candidates
#   - selects best candidate per family
#   - calibrates its decoder automatically
#   - tests + and - angular rates over 3 seeds
#   - checks sensory-population overlap
#   - writes one final summary
# ============================================================


# ----------------------------
# CONFIG
# ----------------------------

CACHE = Path("data/male_cns/cache_v3")

BASE_RATE_HZ = 75.0
DELTA_RATE_HZ = 50.0

OPEN_LOOP_COMMANDS = [-1.0, -0.5, 0.0, 0.5, 1.0]
OPEN_LOOP_TRIALS = 3
OPEN_LOOP_WARMUP_MS = 100.0
OPEN_LOOP_MEASURE_MS = 300.0

TRACE_TAU_MS = 100.0

CAL_WARMUP_MS = 300.0
CAL_SETTLE_MS = 150.0
CAL_MEASURE_MS = 350.0
CAL_REPEATS = 2

CLOSED_LOOP_SECONDS = 4.0
CLOSED_LOOP_SEEDS = [101, 202, 303]
INITIAL_RATE_RAD_S = 1.5

GYRO_TO_HZ = 30.0
MAX_DIFFERENTIAL_HZ = 50.0
MOTOR_GAIN = 0.8

PASS_MIN_REDUCTION_PERCENT = 20.0
PASS_MAX_FAILURES = 0

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/full_validation_v4") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONSOLE_LOG = OUT_DIR / "console.log"
OPEN_LOOP_CSV = OUT_DIR / "open_loop.csv"
CHANNEL_SUMMARY_CSV = OUT_DIR / "channel_summary.csv"
CALIBRATION_CSV = OUT_DIR / "decoder_calibration.csv"
CLOSED_LOOP_CSV = OUT_DIR / "closed_loop.csv"
TRACE_CSV = OUT_DIR / "closed_loop_trace.csv"
OVERLAP_CSV = OUT_DIR / "sensory_overlap.csv"
SUMMARY_TXT = OUT_DIR / "SUMMARY.txt"


# ============================================================
# LOGGING
# ============================================================

class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


_log_file = open(CONSOLE_LOG, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)
sys.stderr = Tee(sys.__stderr__, _log_file)


def header(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def save_csv(path, rows):
    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def norm_type(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(" ", "")


def infer_side(row):
    for column in ("somaSide", "rootSide"):
        value = row.get(column, None)

        if pd.isna(value):
            continue

        value = str(value).strip().upper()

        if value in ("L", "LEFT"):
            return "L"

        if value in ("R", "RIGHT"):
            return "R"

    instance = row.get("instance", "")

    if not pd.isna(instance):
        instance = str(instance).strip().upper()

        if instance.endswith("_L"):
            return "L"

        if instance.endswith("_R"):
            return "R"

    return "?"


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


# ============================================================
# LOAD BRAIN
# ============================================================

header("NEUROFLIGHT FULL AXIS VALIDATION V4")
print("Output:", OUT_DIR)

brain = MaleCNSLIF(
    cache_dir=CACHE,
    dt_ms=0.5,
    global_gain=0.65,
)

metadata = pd.read_feather(
    CACHE / "neurons.feather"
)

body_ids = np.load(
    CACHE / "body_ids.npy",
    mmap_mode="r",
)

body_to_cache = {
    int(body_id): index
    for index, body_id in enumerate(body_ids)
}

metadata["_cache_index"] = metadata["bodyId"].map(body_to_cache)

if metadata["_cache_index"].isna().any():
    raise RuntimeError("metadata/body_ids alignment failed")

metadata["_cache_index"] = metadata["_cache_index"].astype(np.int64)
metadata["_side"] = metadata.apply(infer_side, axis=1)
metadata["_type_clean"] = metadata["type"].map(norm_type)

print("Neurons:", brain.num_neurons)
print("dt:", brain.dt_ms, "ms")


# ============================================================
# POPULATIONS
# ============================================================

def by_types(types, side=None):
    wanted = {norm_type(x) for x in types}

    mask = metadata["_type_clean"].isin(wanted)

    if side is not None:
        mask &= metadata["_side"].eq(side)

    return metadata.loc[
        mask,
        "_cache_index"
    ].to_numpy(dtype=np.int64)


def by_subclass(name, side=None):
    subclass = (
        metadata["subclass"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mask = subclass.eq(
        str(name).strip().lower()
    )

    if side is not None:
        mask &= metadata["_side"].eq(side)

    return metadata.loc[
        mask,
        "_cache_index"
    ].to_numpy(dtype=np.int64)


def make_mask(indices):
    mask = np.zeros(
        brain.num_neurons,
        dtype=bool,
    )

    if len(indices):
        mask[indices] = True

    return mask


haltere_L = by_subclass("haltere", "L")
haltere_R = by_subclass("haltere", "R")

sapp_L = by_types(["SApp"], "L")
sapp_R = by_types(["SApp"], "R")

sn34_L = by_types(["SNpp34"], "L")
sn34_R = by_types(["SNpp34"], "R")
sn25_L = by_types(["SNpp25"], "L")
sn25_R = by_types(["SNpp25"], "R")
sn20_L = by_types(["SNpp20"], "L")
sn20_R = by_types(["SNpp20"], "R")

sn34_all = np.concatenate([sn34_L, sn34_R])
sn25_all = np.concatenate([sn25_L, sn25_R])
sn20_all = np.concatenate([sn20_L, sn20_R])
sn3425_all = np.concatenate([sn34_all, sn25_all])

b1_L = by_types(["b1 MN"], "L")
b1_R = by_types(["b1 MN"], "R")
b2_L = by_types(["b2 MN"], "L")
b2_R = by_types(["b2 MN"], "R")
b3_L = by_types(["b3 MN"], "L")
b3_R = by_types(["b3 MN"], "R")
i1_L = by_types(["i1 MN"], "L")
i1_R = by_types(["i1 MN"], "R")
i2_L = by_types(["i2 MN"], "L")
i2_R = by_types(["i2 MN"], "R")

b12_L = np.concatenate([b1_L, b2_L])
b12_R = np.concatenate([b1_R, b2_R])
b12_all = np.concatenate([b12_L, b12_R])

b3i1_L = np.concatenate([b3_L, i1_L])
b3i1_R = np.concatenate([b3_R, i1_R])
b3i1_all = np.concatenate([b3i1_L, b3i1_R])


header("POPULATION CHECK")

for name, pop in {
    "haltere_L": haltere_L,
    "haltere_R": haltere_R,
    "SApp_L": sapp_L,
    "SApp_R": sapp_R,
    "SNpp34_all": sn34_all,
    "SNpp25_all": sn25_all,
    "SNpp20_all": sn20_all,
    "b12_L": b12_L,
    "b12_R": b12_R,
    "b3i1_L": b3i1_L,
    "b3i1_R": b3i1_R,
}.items():
    print(f"{name:14s}: {len(pop)}")


# ============================================================
# CANDIDATES
# ============================================================

candidates = [
    {
        "name": "roll_all_haltere_b12",
        "family": "roll",
        "pos_sens": haltere_L,
        "neg_sens": haltere_R,
        "pos_motor": b12_L,
        "neg_motor": b12_R,
    },
    {
        "name": "roll_SApp_b12",
        "family": "roll",
        "pos_sens": sapp_L,
        "neg_sens": sapp_R,
        "pos_motor": b12_L,
        "neg_motor": b12_R,
    },
    {
        "name": "pitch_SN3425_vs_SN20",
        "family": "pitch",
        "pos_sens": sn3425_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
    },
    {
        "name": "pitch_SN34_vs_SN20",
        "family": "pitch",
        "pos_sens": sn34_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
    },
    {
        "name": "pitch_SN25_vs_SN20",
        "family": "pitch",
        "pos_sens": sn25_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
    },
    {
        "name": "yaw_SN20_LR_b3i1",
        "family": "yaw",
        "pos_sens": sn20_L,
        "neg_sens": sn20_R,
        "pos_motor": b3i1_L,
        "neg_motor": b3i1_R,
    },
    {
        "name": "yaw_SApp_LR_b3i1",
        "family": "yaw",
        "pos_sens": sapp_L,
        "neg_sens": sapp_R,
        "pos_motor": b3i1_L,
        "neg_motor": b3i1_R,
    },
]


def candidate_by_name(name):
    for c in candidates:
        if c["name"] == name:
            return c
    raise KeyError(name)


# ============================================================
# SENSORY FORCING
# ============================================================

def force_population(rng, population, rate_hz):
    if len(population) == 0:
        return population

    p = float(
        np.clip(
            rate_hz * brain.dt_ms / 1000.0,
            0.0,
            1.0,
        )
    )

    return population[
        rng.random(len(population)) < p
    ]


# ============================================================
# OPEN LOOP SCREEN
# ============================================================

def run_count_window(
    candidate,
    rng,
    command,
    duration_ms,
    count_output,
):
    pos_mask = make_mask(
        candidate["pos_motor"]
    )

    neg_mask = make_mask(
        candidate["neg_motor"]
    )

    rate_pos = float(
        np.clip(
            BASE_RATE_HZ + DELTA_RATE_HZ * command,
            0.0,
            150.0,
        )
    )

    rate_neg = float(
        np.clip(
            BASE_RATE_HZ - DELTA_RATE_HZ * command,
            0.0,
            150.0,
        )
    )

    pos_spikes = 0
    neg_spikes = 0

    steps = int(
        round(
            duration_ms / brain.dt_ms
        )
    )

    for _ in range(steps):
        forced_pos = force_population(
            rng,
            candidate["pos_sens"],
            rate_pos,
        )

        forced_neg = force_population(
            rng,
            candidate["neg_sens"],
            rate_neg,
        )

        forced = np.concatenate([
            forced_pos,
            forced_neg,
        ])

        spikes, _ = brain.step(
            forced_spikes=forced
        )

        if count_output and len(spikes):
            pos_spikes += int(
                pos_mask[spikes].sum()
            )

            neg_spikes += int(
                neg_mask[spikes].sum()
            )

    return pos_spikes - neg_spikes


def open_loop_trial(candidate, command, seed):
    brain.reset()
    rng = np.random.default_rng(seed)

    run_count_window(
        candidate,
        rng,
        0.0,
        OPEN_LOOP_WARMUP_MS,
        False,
    )

    raw = run_count_window(
        candidate,
        rng,
        command,
        OPEN_LOOP_MEASURE_MS,
        True,
    )

    return float(raw)


header("OPEN LOOP SCREEN")

open_rows = []
summary_rows = []

for ci, candidate in enumerate(candidates):
    print()
    print(candidate["name"])

    means = []
    stds = []

    for command in OPEN_LOOP_COMMANDS:
        values = []

        for trial in range(OPEN_LOOP_TRIALS):
            seed = (
                100000
                + ci * 10000
                + int((command + 1.0) * 1000)
                + trial
            )

            raw = open_loop_trial(
                candidate,
                command,
                seed,
            )

            values.append(raw)

            open_rows.append({
                "candidate": candidate["name"],
                "family": candidate["family"],
                "command": command,
                "trial": trial,
                "raw_signal": raw,
            })

        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

        print(
            f"  {command:+.1f} -> "
            f"{means[-1]:+7.2f} +/- {stds[-1]:5.2f}"
        )

    means_arr = np.asarray(means)
    stds_arr = np.asarray(stds)
    commands_arr = np.asarray(OPEN_LOOP_COMMANDS)

    zero_i = OPEN_LOOP_COMMANDS.index(0.0)
    baseline = float(means_arr[zero_i])
    centered = means_arr - baseline

    slope = float(
        np.polyfit(
            commands_arr,
            centered,
            1,
        )[0]
    )

    orientation = (
        1.0
        if slope >= 0.0
        else -1.0
    )

    oriented = centered * orientation

    corr = safe_corr(
        commands_arr,
        oriented,
    )

    endpoint_neg = float(
        oriented[0]
    )

    endpoint_pos = float(
        oriented[-1]
    )

    zero_noise = float(
        stds_arr[zero_i]
    )

    endpoint_min = min(
        abs(endpoint_neg),
        abs(endpoint_pos),
    )

    endpoint_mean = 0.5 * (
        abs(endpoint_neg)
        + abs(endpoint_pos)
    )

    snr = endpoint_min / (
        zero_noise + 1.0
    )

    symmetry = abs(
        abs(endpoint_neg)
        - abs(endpoint_pos)
    ) / (
        endpoint_mean + 1e-9
    )

    sign_ok = (
        endpoint_neg < 0.0
        and endpoint_pos > 0.0
    )

    usable = bool(
        sign_ok
        and corr >= 0.85
        and snr >= 1.5
    )

    score = (
        max(corr, 0.0)
        * snr
        / (1.0 + symmetry)
    )

    summary_rows.append({
        "candidate": candidate["name"],
        "family": candidate["family"],
        "baseline": baseline,
        "orientation": orientation,
        "correlation": corr,
        "endpoint_negative": endpoint_neg,
        "endpoint_positive": endpoint_pos,
        "zero_noise_std": zero_noise,
        "endpoint_snr": snr,
        "symmetry_error": symmetry,
        "usable": usable,
        "score": score,
    })

save_csv(OPEN_LOOP_CSV, open_rows)
save_csv(CHANNEL_SUMMARY_CSV, summary_rows)


best_by_family = {}

for family in ("roll", "pitch", "yaw"):
    rows = [
        row
        for row in summary_rows
        if row["family"] == family
    ]

    rows.sort(
        key=lambda row: row["score"],
        reverse=True,
    )

    if rows:
        best_by_family[family] = rows[0]


header("BEST CHANNELS")

for family, row in best_by_family.items():
    print(
        f"{family.upper():5s}: "
        f"{row['candidate']} "
        f"| corr={row['correlation']:.3f} "
        f"| SNR={row['endpoint_snr']:.2f} "
        f"| score={row['score']:.2f}"
    )


# ============================================================
# CALIBRATED RAW-DIFFERENTIAL DECODER
# ============================================================

trace_decay_brain = math.exp(
    -brain.dt_ms
    / TRACE_TAU_MS
)


def trace_step(
    candidate,
    rng,
    rate_pos,
    rate_neg,
    pos_trace,
    neg_trace,
    pos_mask,
    neg_mask,
):
    forced_pos = force_population(
        rng,
        candidate["pos_sens"],
        rate_pos,
    )

    forced_neg = force_population(
        rng,
        candidate["neg_sens"],
        rate_neg,
    )

    forced = np.concatenate([
        forced_pos,
        forced_neg,
    ])

    spikes, _ = brain.step(
        forced_spikes=forced
    )

    pos_count = 0
    neg_count = 0

    if len(spikes):
        pos_count = int(
            pos_mask[spikes].sum()
        )

        neg_count = int(
            neg_mask[spikes].sum()
        )

    pos_trace = (
        pos_trace
        * trace_decay_brain
        + pos_count
    )

    neg_trace = (
        neg_trace
        * trace_decay_brain
        + neg_count
    )

    return pos_trace, neg_trace


def measure_trace_diff(
    candidate,
    command,
    seed,
):
    brain.reset()
    rng = np.random.default_rng(seed)

    pos_mask = make_mask(
        candidate["pos_motor"]
    )

    neg_mask = make_mask(
        candidate["neg_motor"]
    )

    pos_trace = 0.0
    neg_trace = 0.0

    # Neutral warmup
    warm_steps = int(
        round(
            CAL_WARMUP_MS / brain.dt_ms
        )
    )

    for _ in range(warm_steps):
        pos_trace, neg_trace = trace_step(
            candidate,
            rng,
            BASE_RATE_HZ,
            BASE_RATE_HZ,
            pos_trace,
            neg_trace,
            pos_mask,
            neg_mask,
        )

    rate_pos = float(
        np.clip(
            BASE_RATE_HZ + DELTA_RATE_HZ * command,
            0.0,
            150.0,
        )
    )

    rate_neg = float(
        np.clip(
            BASE_RATE_HZ - DELTA_RATE_HZ * command,
            0.0,
            150.0,
        )
    )

    settle_steps = int(
        round(
            CAL_SETTLE_MS / brain.dt_ms
        )
    )

    for _ in range(settle_steps):
        pos_trace, neg_trace = trace_step(
            candidate,
            rng,
            rate_pos,
            rate_neg,
            pos_trace,
            neg_trace,
            pos_mask,
            neg_mask,
        )

    measure_steps = int(
        round(
            CAL_MEASURE_MS / brain.dt_ms
        )
    )

    diffs = []

    for _ in range(measure_steps):
        pos_trace, neg_trace = trace_step(
            candidate,
            rng,
            rate_pos,
            rate_neg,
            pos_trace,
            neg_trace,
            pos_mask,
            neg_mask,
        )

        diffs.append(
            pos_trace - neg_trace
        )

    return float(
        np.mean(diffs)
    )


header("DECODER CALIBRATION")

calibration_rows = []
calibration_by_family = {}

for fi, family in enumerate(
    ("roll", "pitch", "yaw")
):
    best = best_by_family[family]
    candidate = candidate_by_name(
        best["candidate"]
    )

    command_values = {}

    for command in (-1.0, 0.0, 1.0):
        values = []

        for repeat in range(CAL_REPEATS):
            values.append(
                measure_trace_diff(
                    candidate,
                    command,
                    seed=(
                        700000
                        + fi * 10000
                        + int((command + 1.0) * 1000)
                        + repeat
                    ),
                )
            )

        command_values[command] = float(
            np.mean(values)
        )

    d_minus = command_values[-1.0]
    d_zero = command_values[0.0]
    d_plus = command_values[1.0]

    orientation = (
        1.0
        if d_plus >= d_minus
        else -1.0
    )

    oriented_minus = (
        orientation
        * (d_minus - d_zero)
    )

    oriented_plus = (
        orientation
        * (d_plus - d_zero)
    )

    scale_negative = max(
        abs(oriented_minus),
        1e-6,
    )

    scale_positive = max(
        abs(oriented_plus),
        1e-6,
    )

    valid = bool(
        oriented_minus < 0.0
        and oriented_plus > 0.0
    )

    row = {
        "family": family,
        "candidate": candidate["name"],
        "diff_minus": d_minus,
        "diff_neutral": d_zero,
        "diff_plus": d_plus,
        "orientation": orientation,
        "scale_negative": scale_negative,
        "scale_positive": scale_positive,
        "valid": valid,
    }

    calibration_rows.append(row)
    calibration_by_family[family] = row

    print(
        f"{family.upper():5s} "
        f"| -1={d_minus:+8.3f} "
        f"| 0={d_zero:+8.3f} "
        f"| +1={d_plus:+8.3f} "
        f"| scales={scale_negative:.3f}/{scale_positive:.3f} "
        f"| valid={valid}"
    )

save_csv(CALIBRATION_CSV, calibration_rows)


# ============================================================
# MUJOCO
# ============================================================

header("MUJOCO ISOLATED CLOSED LOOP")

model = mm.load(
    "bitcraze_crazyflie_2"
)

model.opt.gravity[:] = 0.0

gyro_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SENSOR,
    "body_gyro",
)

if gyro_id < 0:
    raise RuntimeError("body_gyro not found")

gyro_adr = int(
    model.sensor_adr[gyro_id]
)

actuator_names = {
    "roll": "x_moment",
    "pitch": "y_moment",
    "yaw": "z_moment",
}

actuator_ids = {}

for family, name in actuator_names.items():
    idx = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if idx < 0:
        raise RuntimeError(
            f"{name} actuator not found"
        )

    actuator_ids[family] = idx


physics_dt = float(
    model.opt.timestep
)

lif_steps_per_physics = max(
    1,
    int(
        round(
            physics_dt * 1000.0
            / brain.dt_ms
        )
    ),
)


def gyro(data):
    return np.asarray(
        data.sensordata[
            gyro_adr:gyro_adr + 3
        ],
        dtype=float,
    ).copy()


def neutral_state(
    candidate,
    seed,
):
    brain.reset()
    rng = np.random.default_rng(seed)

    pos_mask = make_mask(
        candidate["pos_motor"]
    )

    neg_mask = make_mask(
        candidate["neg_motor"]
    )

    pos_trace = 0.0
    neg_trace = 0.0

    steps = int(
        round(
            CAL_WARMUP_MS / brain.dt_ms
        )
    )

    for _ in range(steps):
        pos_trace, neg_trace = trace_step(
            candidate,
            rng,
            BASE_RATE_HZ,
            BASE_RATE_HZ,
            pos_trace,
            neg_trace,
            pos_mask,
            neg_mask,
        )

    return (
        rng,
        pos_mask,
        neg_mask,
        pos_trace,
        neg_trace,
    )


def decode_diff(
    diff,
    calibration,
):
    oriented = (
        calibration["orientation"]
        * (
            diff
            - calibration["diff_neutral"]
        )
    )

    if oriented >= 0.0:
        scale = calibration[
            "scale_positive"
        ]
    else:
        scale = calibration[
            "scale_negative"
        ]

    return float(
        np.clip(
            oriented / max(scale, 1e-6),
            -1.0,
            1.0,
        )
    )


def run_closed_loop(
    family,
    candidate,
    calibration,
    initial_rate,
    seed,
):
    axis = {
        "roll": 0,
        "pitch": 1,
        "yaw": 2,
    }[family]

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = [
        0.0,
        0.0,
        0.5,
    ]

    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    data.qvel[:] = 0.0
    data.qvel[3 + axis] = initial_rate
    data.ctrl[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    (
        rng,
        pos_mask,
        neg_mask,
        pos_trace,
        neg_trace,
    ) = neutral_state(
        candidate,
        seed,
    )

    total_steps = int(
        round(
            CLOSED_LOOP_SECONDS
            / physics_dt
        )
    )

    log_every = max(
        1,
        int(
            round(
                0.1 / physics_dt
            )
        ),
    )

    trace_rows = []

    max_abs = abs(initial_rate)
    control_abs_sum = 0.0
    control_sum = 0.0

    for step in range(total_steps):
        g = gyro(data)
        angular_rate = float(g[axis])

        max_abs = max(
            max_abs,
            abs(angular_rate),
        )

        differential = float(
            np.clip(
                GYRO_TO_HZ * angular_rate,
                -MAX_DIFFERENTIAL_HZ,
                MAX_DIFFERENTIAL_HZ,
            )
        )

        rate_pos = float(
            np.clip(
                BASE_RATE_HZ + differential,
                0.0,
                150.0,
            )
        )

        rate_neg = float(
            np.clip(
                BASE_RATE_HZ - differential,
                0.0,
                150.0,
            )
        )

        for _ in range(
            lif_steps_per_physics
        ):
            pos_trace, neg_trace = trace_step(
                candidate,
                rng,
                rate_pos,
                rate_neg,
                pos_trace,
                neg_trace,
                pos_mask,
                neg_mask,
            )

        diff = (
            pos_trace
            - neg_trace
        )

        brain_signal = decode_diff(
            diff,
            calibration,
        )

        control = float(
            np.clip(
                MOTOR_GAIN * brain_signal,
                -1.0,
                1.0,
            )
        )

        data.ctrl[:] = 0.0
        data.ctrl[
            actuator_ids[family]
        ] = control

        mujoco.mj_step(
            model,
            data,
        )

        control_abs_sum += abs(control)
        control_sum += control

        if step % log_every == 0:
            trace_rows.append({
                "family": family,
                "seed": seed,
                "initial_rate": initial_rate,
                "time": step * physics_dt,
                "gyro": angular_rate,
                "rate_pos": rate_pos,
                "rate_neg": rate_neg,
                "motor_diff": diff,
                "brain_signal": brain_signal,
                "control": control,
            })

    final_rate = float(
        gyro(data)[axis]
    )

    reduction = (
        1.0
        - abs(final_rate)
        / max(abs(initial_rate), 1e-9)
    ) * 100.0

    stable = bool(
        abs(final_rate) < abs(initial_rate)
        and max_abs < 1.5 * abs(initial_rate)
    )

    return (
        {
            "family": family,
            "candidate": candidate["name"],
            "seed": seed,
            "initial_rate": initial_rate,
            "final_rate": final_rate,
            "reduction_percent": reduction,
            "max_abs_rate": max_abs,
            "mean_control": control_sum / total_steps,
            "mean_abs_control": control_abs_sum / total_steps,
            "stable": stable,
        },
        trace_rows,
    )


closed_rows = []
trace_rows = []

for family_index, family in enumerate(
    ("roll", "pitch", "yaw")
):
    best = best_by_family[family]
    candidate = candidate_by_name(
        best["candidate"]
    )

    calibration = calibration_by_family[
        family
    ]

    if not calibration["valid"]:
        print(
            family.upper(),
            "SKIPPED: invalid calibration",
        )
        continue

    print()
    print(
        family.upper(),
        "->",
        candidate["name"],
    )

    for direction_index, initial_rate in enumerate(
        (+INITIAL_RATE_RAD_S, -INITIAL_RATE_RAD_S)
    ):
        for seed in CLOSED_LOOP_SEEDS:
            result, rows = run_closed_loop(
                family,
                candidate,
                calibration,
                initial_rate,
                seed + 10000 * family_index + 1000 * direction_index,
            )

            closed_rows.append(result)
            trace_rows.extend(rows)

            print(
                f"  seed={seed:3d} "
                f"{initial_rate:+.2f} "
                f"-> {result['final_rate']:+.4f} "
                f"| red={result['reduction_percent']:5.1f}% "
                f"| ctrl={result['mean_abs_control']:.3f} "
                f"| stable={result['stable']}"
            )

save_csv(CLOSED_LOOP_CSV, closed_rows)
save_csv(TRACE_CSV, trace_rows)


# ============================================================
# SENSORY OVERLAP
#
# This is important before any 3-axis combined simulation.
# If two isolated axes reuse the same sensory neurons, they are
# not independent input channels yet.
# ============================================================

header("BEST-CHANNEL SENSORY OVERLAP")

overlap_rows = []

families = [
    family
    for family in ("roll", "pitch", "yaw")
    if family in best_by_family
]

best_unions = {}

for family in families:
    candidate = candidate_by_name(
        best_by_family[family]["candidate"]
    )

    best_unions[family] = set(
        int(x)
        for x in np.concatenate([
            candidate["pos_sens"],
            candidate["neg_sens"],
        ])
    )

for i, a in enumerate(families):
    for b in families[i + 1:]:
        A = best_unions[a]
        B = best_unions[b]

        inter = len(A & B)
        union = len(A | B)

        jaccard = (
            inter / union
            if union
            else 0.0
        )

        smaller_fraction = (
            inter / min(len(A), len(B))
            if min(len(A), len(B)) > 0
            else 0.0
        )

        row = {
            "family_a": a,
            "family_b": b,
            "size_a": len(A),
            "size_b": len(B),
            "intersection": inter,
            "jaccard": jaccard,
            "fraction_of_smaller": smaller_fraction,
        }

        overlap_rows.append(row)

        print(
            f"{a:5s} vs {b:5s}: "
            f"intersection={inter:3d} "
            f"| Jaccard={jaccard:.3f} "
            f"| smaller covered={smaller_fraction:.3f}"
        )

save_csv(OVERLAP_CSV, overlap_rows)


# ============================================================
# FINAL STATISTICS
# ============================================================

header("FINAL VERDICT")

verdicts = {}
summary_lines = []

summary_lines.append(
    "NEUROFLIGHT FULL AXIS VALIDATION V4"
)

summary_lines.append(
    f"Timestamp: {timestamp}"
)

summary_lines.append("")
summary_lines.append(
    "The V4 decoder uses automatically calibrated RAW motor differential."
)

summary_lines.append(
    "It does not use the saturating (L-R)/(L+R) decoder from V3."
)

summary_lines.append("")

for family in ("roll", "pitch", "yaw"):
    best = best_by_family.get(family)
    calibration = calibration_by_family.get(family)

    rows = [
        row
        for row in closed_rows
        if row["family"] == family
    ]

    summary_lines.append(
        f"{family.upper()}:"
    )

    if best is None:
        summary_lines.append(
            "  no open-loop candidate"
        )
        verdicts[family] = False
        continue

    summary_lines.append(
        f"  candidate: {best['candidate']}"
    )

    summary_lines.append(
        f"  open-loop corr: {best['correlation']:.3f}"
    )

    summary_lines.append(
        f"  open-loop SNR: {best['endpoint_snr']:.2f}"
    )

    if calibration:
        summary_lines.append(
            "  calibration: "
            f"-={calibration['diff_minus']:+.3f}, "
            f"0={calibration['diff_neutral']:+.3f}, "
            f"+={calibration['diff_plus']:+.3f}"
        )

    family_failures = 0

    reductions = []

    for direction in (
        +INITIAL_RATE_RAD_S,
        -INITIAL_RATE_RAD_S,
    ):
        dir_rows = [
            row
            for row in rows
            if row["initial_rate"] == direction
        ]

        if dir_rows:
            final_abs = np.asarray(
                [
                    abs(row["final_rate"])
                    for row in dir_rows
                ]
            )

            reduction_arr = np.asarray(
                [
                    row["reduction_percent"]
                    for row in dir_rows
                ]
            )

            stable_flags = [
                bool(row["stable"])
                for row in dir_rows
            ]

            reductions.extend(
                reduction_arr.tolist()
            )

            direction_ok = bool(
                np.mean(reduction_arr)
                >= PASS_MIN_REDUCTION_PERCENT
                and all(stable_flags)
            )

            if not direction_ok:
                family_failures += 1

            summary_lines.append(
                f"  {direction:+.2f} rad/s -> "
                f"mean |final|={np.mean(final_abs):.4f}, "
                f"mean reduction={np.mean(reduction_arr):.1f}%, "
                f"range={np.min(reduction_arr):.1f}..{np.max(reduction_arr):.1f}%, "
                f"PASS={direction_ok}"
            )
        else:
            family_failures += 1
            summary_lines.append(
                f"  {direction:+.2f}: no closed-loop result"
            )

    family_ok = bool(
        family_failures
        <= PASS_MAX_FAILURES
    )

    verdicts[family] = family_ok

    summary_lines.append(
        f"  ISOLATED AXIS VERDICT: "
        f"{'PASS' if family_ok else 'NOT PASS'}"
    )

    summary_lines.append("")


summary_lines.append(
    "SENSORY-INDEPENDENCE CHECK:"
)

overlap_warning = False

for row in overlap_rows:
    summary_lines.append(
        "  "
        f"{row['family_a']} vs {row['family_b']}: "
        f"{row['intersection']} shared neurons, "
        f"{100.0 * row['fraction_of_smaller']:.1f}% of the smaller channel"
    )

    if row["fraction_of_smaller"] > 0.25:
        overlap_warning = True


summary_lines.append("")

if all(
    verdicts.get(family, False)
    for family in ("roll", "pitch", "yaw")
):
    summary_lines.append(
        "ISOLATED 3-AXIS RESULT: PASS"
    )
else:
    summary_lines.append(
        "ISOLATED 3-AXIS RESULT: NOT PASS"
    )


if overlap_warning:
    summary_lines.append(
        "COMBINED 3-AXIS RESULT: NOT YET VALIDATED."
    )

    summary_lines.append(
        "Reason: at least two selected axis encoders share a substantial"
    )

    summary_lines.append(
        "fraction of the same sensory neurons, so independent simultaneous"
    )

    summary_lines.append(
        "roll/pitch/yaw commands cannot yet be assumed."
    )
else:
    summary_lines.append(
        "The selected sensory channels have low overlap; a combined test is reasonable next."
    )


summary_lines.append("")
summary_lines.append(
    "SCIENTIFIC LIMITATION:"
)

summary_lines.append(
    "These tests validate a computational control interface built from"
)

summary_lines.append(
    "MaleCNS topology, assumed LIF dynamics, synthetic sensory forcing,"
)

summary_lines.append(
    "and a fly-motor-to-drone actuator adapter. They do not establish"
)

summary_lines.append(
    "that the selected channels are literal biological roll/pitch/yaw axes."
)

summary = "\n".join(
    summary_lines
)

print(summary)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8",
) as f:
    f.write(
        summary + "\n"
    )


print()
print("=" * 78)
print("FILES TO SEND BACK")
print("=" * 78)
print(SUMMARY_TXT)
print(CHANNEL_SUMMARY_CSV)
print(CALIBRATION_CSV)
print(CLOSED_LOOP_CSV)
print(OVERLAP_CSV)
print(CONSOLE_LOG)
