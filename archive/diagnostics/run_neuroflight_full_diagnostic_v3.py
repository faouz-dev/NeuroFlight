from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import math
import sys
import traceback

import mujoco
import mujoco_menagerie as mm
import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


# ============================================================
# NEUROFLIGHT - FULL DIAGNOSTIC SUITE V3
#
# Goal:
#   1) verify cache / populations / direct connectivity
#   2) screen several roll / pitch / yaw candidate channels
#   3) run isolated closed-loop damping tests in MuJoCo
#   4) save EVERYTHING to one timestamped log folder
#
# IMPORTANT:
# This tests a computational MaleCNS + synthetic sensory encoding.
# It does NOT prove that a candidate channel is the literal biological
# roll/pitch/yaw axis of a real fly.
# ============================================================


# ----------------------------
# CONFIG
# ----------------------------

CACHE = Path("data/male_cns/cache_v3")

OPEN_LOOP_COMMANDS = [-1.0, -0.5, 0.0, 0.5, 1.0]
OPEN_LOOP_TRIALS = 3
OPEN_LOOP_WARMUP_MS = 100.0
OPEN_LOOP_MEASURE_MS = 300.0

BASE_RATE_HZ = 75.0
DELTA_RATE_HZ = 50.0

CLOSED_LOOP_WARMUP_MS = 500.0
CLOSED_LOOP_SECONDS = 4.0
INITIAL_RATE_RAD_S = 1.5
GYRO_TO_HZ = 30.0
MAX_DIFFERENTIAL_HZ = 50.0
TRACE_TAU_MS = 100.0
MOTOR_GAIN = 0.8

# Minimum open-loop quality required before a channel is used
# automatically in closed-loop.
MIN_CORRELATION_FOR_CLOSED_LOOP = 0.70

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/full_diagnostic") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONSOLE_LOG = OUT_DIR / "console.log"
DIRECT_CSV = OUT_DIR / "direct_connectivity.csv"
RESPONSES_CSV = OUT_DIR / "channel_responses.csv"
CHANNEL_SUMMARY_CSV = OUT_DIR / "channel_summary.csv"
CLOSED_LOOP_CSV = OUT_DIR / "closed_loop.csv"
SUMMARY_TXT = OUT_DIR / "SUMMARY.txt"
MANIFEST_JSON = OUT_DIR / "manifest.json"


# ============================================================
# TEE CONSOLE -> FILE
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


# ============================================================
# HELPERS
# ============================================================

def norm_type(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace(" ", "")


def infer_side(row) -> str:
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


def safe_corr(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return 0.0

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0

    return float(np.corrcoef(x, y)[0, 1])


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return

    if fieldnames is None:
        keys = []
        seen = set()

        for row in rows:
            for key in row.keys():
                if key not in seen:
                    keys.append(key)
                    seen.add(key)

        fieldnames = keys

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_header(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# LOAD MALE CNS
# ============================================================

print_header("NEUROFLIGHT FULL DIAGNOSTIC V3")
print("Output folder:", OUT_DIR)

print()
print("Loading MaleCNS LIF...")

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

row_ptr = np.load(
    CACHE / "row_ptr.npy",
    mmap_mode="r",
)

post_idx = np.load(
    CACHE / "post_idx.npy",
    mmap_mode="r",
)

syn_count = np.load(
    CACHE / "syn_count.npy",
    mmap_mode="r",
)

print("Neurons    :", brain.num_neurons)
print("Edges      :", len(post_idx))
print("dt         :", brain.dt_ms, "ms")


# ============================================================
# ALIGN METADATA TO CACHE INDICES
# ============================================================

body_to_cache = {
    int(body_id): index
    for index, body_id in enumerate(body_ids)
}

metadata["_cache_index"] = metadata["bodyId"].map(body_to_cache)

if metadata["_cache_index"].isna().any():
    missing = int(metadata["_cache_index"].isna().sum())
    raise RuntimeError(
        f"{missing} metadata rows are not present in body_ids.npy"
    )

metadata["_cache_index"] = metadata["_cache_index"].astype(np.int64)
metadata["_side"] = metadata.apply(infer_side, axis=1)
metadata["_type_clean"] = metadata["type"].map(norm_type)


# ============================================================
# POPULATION HELPERS
# ============================================================

def population_by_types(types, side=None) -> np.ndarray:
    clean_targets = {
        norm_type(t)
        for t in types
    }

    mask = metadata["_type_clean"].isin(clean_targets)

    if side is not None:
        mask &= metadata["_side"].eq(side)

    return metadata.loc[
        mask,
        "_cache_index"
    ].to_numpy(dtype=np.int64)


def population_by_subclass(subclass, side=None) -> np.ndarray:
    values = (
        metadata["subclass"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mask = values.eq(str(subclass).strip().lower())

    if side is not None:
        mask &= metadata["_side"].eq(side)

    return metadata.loc[
        mask,
        "_cache_index"
    ].to_numpy(dtype=np.int64)


def mask_for(indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(
        brain.num_neurons,
        dtype=bool,
    )

    if len(indices):
        mask[indices] = True

    return mask


# ============================================================
# IMPORTANT POPULATIONS
# ============================================================

haltere_L = population_by_subclass("haltere", "L")
haltere_R = population_by_subclass("haltere", "R")

sapp_L = population_by_types(["SApp"], "L")
sapp_R = population_by_types(["SApp"], "R")

sn34_L = population_by_types(["SNpp34"], "L")
sn34_R = population_by_types(["SNpp34"], "R")
sn25_L = population_by_types(["SNpp25"], "L")
sn25_R = population_by_types(["SNpp25"], "R")
sn20_L = population_by_types(["SNpp20"], "L")
sn20_R = population_by_types(["SNpp20"], "R")

sn3425_all = np.concatenate([
    sn34_L, sn34_R,
    sn25_L, sn25_R,
])

sn34_all = np.concatenate([
    sn34_L, sn34_R,
])

sn25_all = np.concatenate([
    sn25_L, sn25_R,
])

sn20_all = np.concatenate([
    sn20_L, sn20_R,
])

b1_L = population_by_types(["b1 MN"], "L")
b1_R = population_by_types(["b1 MN"], "R")
b2_L = population_by_types(["b2 MN"], "L")
b2_R = population_by_types(["b2 MN"], "R")
b3_L = population_by_types(["b3 MN"], "L")
b3_R = population_by_types(["b3 MN"], "R")
i1_L = population_by_types(["i1 MN"], "L")
i1_R = population_by_types(["i1 MN"], "R")
i2_L = population_by_types(["i2 MN"], "L")
i2_R = population_by_types(["i2 MN"], "R")

b12_L = np.concatenate([b1_L, b2_L])
b12_R = np.concatenate([b1_R, b2_R])

b12_all = np.concatenate([
    b1_L, b1_R,
    b2_L, b2_R,
])

b3i1_L = np.concatenate([
    b3_L, i1_L,
])

b3i1_R = np.concatenate([
    b3_R, i1_R,
])

b3i1_all = np.concatenate([
    b3_L, b3_R,
    i1_L, i1_R,
])

b3i1i2_all = np.concatenate([
    b3_L, b3_R,
    i1_L, i1_R,
    i2_L, i2_R,
])


print_header("POPULATIONS")

population_report = {
    "haltere_L": len(haltere_L),
    "haltere_R": len(haltere_R),
    "SApp_L": len(sapp_L),
    "SApp_R": len(sapp_R),
    "SNpp34_L": len(sn34_L),
    "SNpp34_R": len(sn34_R),
    "SNpp25_L": len(sn25_L),
    "SNpp25_R": len(sn25_R),
    "SNpp20_L": len(sn20_L),
    "SNpp20_R": len(sn20_R),
    "b1_L": len(b1_L),
    "b1_R": len(b1_R),
    "b2_L": len(b2_L),
    "b2_R": len(b2_R),
    "b3_L": len(b3_L),
    "b3_R": len(b3_R),
    "i1_L": len(i1_L),
    "i1_R": len(i1_R),
    "i2_L": len(i2_L),
    "i2_R": len(i2_R),
}

for name, count in population_report.items():
    print(f"{name:12s}: {count}")


# ============================================================
# DIRECT CONNECTIVITY MATRIX
# ============================================================

print_header("DIRECT HALTERE -> FLIGHT MOTOR CONNECTIVITY")

motor_targets = {
    "b1_L": b1_L,
    "b1_R": b1_R,
    "b2_L": b2_L,
    "b2_R": b2_R,
    "b3_L": b3_L,
    "b3_R": b3_R,
    "i1_L": i1_L,
    "i1_R": i1_R,
    "i2_L": i2_L,
    "i2_R": i2_R,
}

motor_target_sets = {
    name: set(int(x) for x in indices.tolist())
    for name, indices in motor_targets.items()
}

haltere_rows = metadata[
    metadata["subclass"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("haltere")
].copy()

haltere_rows["_type_display"] = (
    haltere_rows["type"]
    .fillna("")
    .astype(str)
    .str.strip()
)

haltere_rows.loc[
    haltere_rows["_type_display"].eq(""),
    "_type_display"
] = "UNLABELED"

direct_rows = []

for (type_name, side), group in haltere_rows.groupby(
    ["_type_display", "_side"]
):
    if side not in ("L", "R"):
        continue

    pre_indices = group["_cache_index"].to_numpy(dtype=np.int64)

    result = {
        "haltere_type": type_name,
        "haltere_side": side,
        "n_neurons": len(pre_indices),
    }

    for motor_name in motor_targets:
        result[motor_name] = 0

    for pre in pre_indices:
        start = int(row_ptr[pre])
        end = int(row_ptr[pre + 1])

        targets = post_idx[start:end]
        weights = syn_count[start:end]

        for target_idx, weight in zip(targets, weights):
            target_idx = int(target_idx)
            weight = int(weight)

            for motor_name, target_set in motor_target_sets.items():
                if target_idx in target_set:
                    result[motor_name] += weight

    result["total_direct"] = sum(
        result[name]
        for name in motor_targets
    )

    result["direct_per_neuron"] = (
        result["total_direct"]
        / max(result["n_neurons"], 1)
    )

    direct_rows.append(result)

direct_rows.sort(
    key=lambda row: row["total_direct"],
    reverse=True,
)

save_csv(DIRECT_CSV, direct_rows)

for row in direct_rows[:15]:
    print(
        f"{row['haltere_type']:10s} "
        f"{row['haltere_side']} | "
        f"n={row['n_neurons']:3d} | "
        f"direct={row['total_direct']:4d}"
    )


# ============================================================
# CANDIDATE CHANNELS
#
# pos_sens / neg_sens:
#     + command raises pos_sens rate and lowers neg_sens rate.
#
# pos_motor / neg_motor:
#     raw motor signal = spikes(pos_motor) - spikes(neg_motor)
#
# The suite is allowed to flip decoder_sign automatically if
# the measured response is inverted.
# ============================================================

candidates = [
    {
        "name": "roll_all_haltere_b12",
        "family": "roll",
        "pos_sens": haltere_L,
        "neg_sens": haltere_R,
        "pos_motor": b12_L,
        "neg_motor": b12_R,
        "description": "all haltere L/R -> b1+b2 L/R",
    },
    {
        "name": "roll_SApp_b12",
        "family": "roll",
        "pos_sens": sapp_L,
        "neg_sens": sapp_R,
        "pos_motor": b12_L,
        "neg_motor": b12_R,
        "description": "SApp L/R -> b1+b2 L/R",
    },
    {
        "name": "pitch_SN3425_vs_SN20",
        "family": "pitch",
        "pos_sens": sn3425_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
        "description": "SNpp34+25 vs SNpp20 -> b1+b2 vs b3+i1",
    },
    {
        "name": "pitch_SN34_vs_SN20",
        "family": "pitch",
        "pos_sens": sn34_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
        "description": "SNpp34 vs SNpp20 -> b1+b2 vs b3+i1",
    },
    {
        "name": "pitch_SN25_vs_SN20",
        "family": "pitch",
        "pos_sens": sn25_all,
        "neg_sens": sn20_all,
        "pos_motor": b12_all,
        "neg_motor": b3i1_all,
        "description": "SNpp25 vs SNpp20 -> b1+b2 vs b3+i1",
    },
    {
        "name": "yaw_SN20_LR_b3i1",
        "family": "yaw",
        "pos_sens": sn20_L,
        "neg_sens": sn20_R,
        "pos_motor": b3i1_L,
        "neg_motor": b3i1_R,
        "description": "SNpp20 L/R -> b3+i1 L/R",
    },
    {
        "name": "yaw_SApp_LR_b3i1",
        "family": "yaw",
        "pos_sens": sapp_L,
        "neg_sens": sapp_R,
        "pos_motor": b3i1_L,
        "neg_motor": b3i1_R,
        "description": "SApp L/R -> b3+i1 L/R",
    },
]


# ============================================================
# OPEN LOOP
# ============================================================

def run_open_loop_trial(candidate, command, seed):
    brain.reset()

    rng = np.random.default_rng(seed)

    pos_sens = candidate["pos_sens"]
    neg_sens = candidate["neg_sens"]

    pos_motor_mask = mask_for(candidate["pos_motor"])
    neg_motor_mask = mask_for(candidate["neg_motor"])

    def stimulate(rate_pos, rate_neg, duration_ms, count_output):
        steps = int(round(duration_ms / brain.dt_ms))

        forced_pos_total = 0
        forced_neg_total = 0
        pos_motor_spikes = 0
        neg_motor_spikes = 0

        p_pos = np.clip(
            rate_pos * brain.dt_ms / 1000.0,
            0.0,
            1.0,
        )

        p_neg = np.clip(
            rate_neg * brain.dt_ms / 1000.0,
            0.0,
            1.0,
        )

        for _ in range(steps):
            forced_pos = pos_sens[
                rng.random(len(pos_sens)) < p_pos
            ]

            forced_neg = neg_sens[
                rng.random(len(neg_sens)) < p_neg
            ]

            forced = np.concatenate([
                forced_pos,
                forced_neg,
            ])

            spikes, _ = brain.step(
                forced_spikes=forced
            )

            if count_output:
                forced_pos_total += len(forced_pos)
                forced_neg_total += len(forced_neg)

                if len(spikes):
                    pos_motor_spikes += int(
                        pos_motor_mask[spikes].sum()
                    )

                    neg_motor_spikes += int(
                        neg_motor_mask[spikes].sum()
                    )

        return (
            forced_pos_total,
            forced_neg_total,
            pos_motor_spikes,
            neg_motor_spikes,
        )

    # Warmup with neutral sensory rates.
    stimulate(
        BASE_RATE_HZ,
        BASE_RATE_HZ,
        OPEN_LOOP_WARMUP_MS,
        False,
    )

    rate_pos = float(np.clip(
        BASE_RATE_HZ + DELTA_RATE_HZ * command,
        0.0,
        150.0,
    ))

    rate_neg = float(np.clip(
        BASE_RATE_HZ - DELTA_RATE_HZ * command,
        0.0,
        150.0,
    ))

    (
        forced_pos,
        forced_neg,
        motor_pos,
        motor_neg,
    ) = stimulate(
        rate_pos,
        rate_neg,
        OPEN_LOOP_MEASURE_MS,
        True,
    )

    raw_signal = motor_pos - motor_neg

    return {
        "candidate": candidate["name"],
        "family": candidate["family"],
        "command": command,
        "trial_seed": seed,
        "rate_pos_hz": rate_pos,
        "rate_neg_hz": rate_neg,
        "forced_pos": forced_pos,
        "forced_neg": forced_neg,
        "motor_pos": motor_pos,
        "motor_neg": motor_neg,
        "raw_signal": raw_signal,
    }


print_header("OPEN LOOP CHANNEL SCREEN")

response_rows = []

for candidate_index, candidate in enumerate(candidates):
    print()
    print(candidate["name"])
    print(" ", candidate["description"])
    print(
        "  sensory sizes:",
        len(candidate["pos_sens"]),
        "/",
        len(candidate["neg_sens"]),
        "| motor sizes:",
        len(candidate["pos_motor"]),
        "/",
        len(candidate["neg_motor"]),
    )

    if (
        len(candidate["pos_sens"]) == 0
        or len(candidate["neg_sens"]) == 0
        or len(candidate["pos_motor"]) == 0
        or len(candidate["neg_motor"]) == 0
    ):
        print("  SKIP: empty population")
        continue

    for command in OPEN_LOOP_COMMANDS:
        trial_values = []

        for trial in range(OPEN_LOOP_TRIALS):
            seed = (
                100000
                + candidate_index * 10000
                + int((command + 1.0) * 1000)
                + trial
            )

            row = run_open_loop_trial(
                candidate,
                command,
                seed,
            )

            response_rows.append(row)
            trial_values.append(row["raw_signal"])

        print(
            f"  command={command:+.1f} "
            f"raw={np.mean(trial_values):+7.2f} "
            f"+/- {np.std(trial_values):5.2f}"
        )

save_csv(RESPONSES_CSV, response_rows)


# ============================================================
# OPEN LOOP METRICS
# ============================================================

print_header("CHANNEL METRICS")

channel_summary_rows = []

for candidate in candidates:
    rows = [
        row
        for row in response_rows
        if row["candidate"] == candidate["name"]
    ]

    if not rows:
        continue

    means = []
    stds = []

    for command in OPEN_LOOP_COMMANDS:
        values = np.asarray(
            [
                row["raw_signal"]
                for row in rows
                if row["command"] == command
            ],
            dtype=float,
        )

        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)))

    commands = np.asarray(
        OPEN_LOOP_COMMANDS,
        dtype=float,
    )

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    zero_index = OPEN_LOOP_COMMANDS.index(0.0)
    baseline = float(means[zero_index])

    centered = means - baseline

    slope = float(
        np.polyfit(
            commands,
            centered,
            deg=1,
        )[0]
    )

    decoder_sign = 1.0 if slope >= 0.0 else -1.0

    oriented = centered * decoder_sign

    corr = safe_corr(
        commands,
        oriented,
    )

    low = float(oriented[0])
    high = float(oriented[-1])

    zero_noise = float(stds[zero_index])

    endpoint_min = min(
        abs(low),
        abs(high),
    )

    endpoint_mean = 0.5 * (
        abs(low)
        + abs(high)
    )

    symmetry = abs(
        abs(low)
        - abs(high)
    ) / (endpoint_mean + 1e-9)

    sign_ok = (
        low < 0.0
        and high > 0.0
    )

    snr = endpoint_min / (
        zero_noise + 1.0
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

    result = {
        "candidate": candidate["name"],
        "family": candidate["family"],
        "description": candidate["description"],
        "baseline_raw": baseline,
        "decoder_sign": decoder_sign,
        "correlation": corr,
        "slope": slope,
        "endpoint_negative": low,
        "endpoint_positive": high,
        "zero_noise_std": zero_noise,
        "endpoint_snr": snr,
        "symmetry_error": symmetry,
        "usable": usable,
        "score": score,
    }

    channel_summary_rows.append(result)

    print(
        f"{candidate['name']:28s} "
        f"| corr={corr:5.3f} "
        f"| SNR={snr:5.2f} "
        f"| sym={symmetry:5.2f} "
        f"| sign={decoder_sign:+.0f} "
        f"| usable={usable} "
        f"| score={score:6.2f}"
    )

save_csv(CHANNEL_SUMMARY_CSV, channel_summary_rows)


# ============================================================
# BEST CANDIDATE PER FAMILY
# ============================================================

best_by_family = {}

for family in ("roll", "pitch", "yaw"):
    rows = [
        row
        for row in channel_summary_rows
        if row["family"] == family
    ]

    if rows:
        rows.sort(
            key=lambda row: row["score"],
            reverse=True,
        )

        best_by_family[family] = rows[0]


print_header("BEST OPEN LOOP CANDIDATES")

for family, row in best_by_family.items():
    print(
        f"{family.upper():5s}: "
        f"{row['candidate']} "
        f"| corr={row['correlation']:.3f} "
        f"| SNR={row['endpoint_snr']:.2f} "
        f"| score={row['score']:.2f}"
    )


# ============================================================
# MUJOCO
# ============================================================

print_header("MUJOCO CLOSED LOOP")

model = mm.load(
    "bitcraze_crazyflie_2"
)

# Isolate rotational behavior.
model.opt.gravity[:] = 0.0

gyro_sensor_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SENSOR,
    "body_gyro",
)

if gyro_sensor_id < 0:
    raise RuntimeError(
        "body_gyro sensor not found"
    )

gyro_adr = int(
    model.sensor_adr[
        gyro_sensor_id
    ]
)

gyro_dim = int(
    model.sensor_dim[
        gyro_sensor_id
    ]
)

if gyro_dim < 3:
    raise RuntimeError(
        "body_gyro has fewer than 3 dimensions"
    )

actuator_names = {
    "roll": "x_moment",
    "pitch": "y_moment",
    "yaw": "z_moment",
}

actuator_ids = {}

for family, actuator_name in actuator_names.items():
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator {actuator_name} not found"
        )

    actuator_ids[family] = actuator_id

physics_dt = float(
    model.opt.timestep
)

physics_dt_ms = physics_dt * 1000.0

lif_steps_per_physics = max(
    1,
    int(
        round(
            physics_dt_ms
            / brain.dt_ms
        )
    ),
)

trace_decay = math.exp(
    -physics_dt_ms
    / TRACE_TAU_MS
)


def get_gyro(data) -> np.ndarray:
    return np.asarray(
        data.sensordata[
            gyro_adr:gyro_adr + 3
        ],
        dtype=float,
    ).copy()


def find_candidate(name):
    for candidate in candidates:
        if candidate["name"] == name:
            return candidate

    raise KeyError(name)


def neutral_warmup(
    candidate,
    rng,
    pos_motor_mask,
    neg_motor_mask,
):
    brain.reset()

    pos_trace = 0.0
    neg_trace = 0.0

    p = (
        BASE_RATE_HZ
        * brain.dt_ms
        / 1000.0
    )

    brain_steps = int(
        round(
            CLOSED_LOOP_WARMUP_MS
            / brain.dt_ms
        )
    )

    for _ in range(brain_steps):
        forced_pos = candidate["pos_sens"][
            rng.random(
                len(candidate["pos_sens"])
            ) < p
        ]

        forced_neg = candidate["neg_sens"][
            rng.random(
                len(candidate["neg_sens"])
            ) < p
        ]

        forced = np.concatenate([
            forced_pos,
            forced_neg,
        ])

        spikes, _ = brain.step(
            forced_spikes=forced
        )

        pos_spikes = 0
        neg_spikes = 0

        if len(spikes):
            pos_spikes = int(
                pos_motor_mask[
                    spikes
                ].sum()
            )

            neg_spikes = int(
                neg_motor_mask[
                    spikes
                ].sum()
            )

        # Brain dt trace update during warmup.
        warm_decay = math.exp(
            -brain.dt_ms
            / TRACE_TAU_MS
        )

        pos_trace = (
            pos_trace
            * warm_decay
            + pos_spikes
        )

        neg_trace = (
            neg_trace
            * warm_decay
            + neg_spikes
        )

    baseline_ratio = (
        pos_trace
        - neg_trace
    ) / (
        pos_trace
        + neg_trace
        + 1e-6
    )

    return (
        pos_trace,
        neg_trace,
        baseline_ratio,
    )


def run_closed_loop(
    family,
    candidate,
    decoder_sign,
    initial_rate,
    seed,
):
    axis_index = {
        "roll": 0,
        "pitch": 1,
        "yaw": 2,
    }[family]

    qvel_index = 3 + axis_index

    actuator_id = actuator_ids[
        family
    ]

    data = mujoco.MjData(
        model
    )

    mujoco.mj_resetData(
        model,
        data,
    )

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
    data.qvel[qvel_index] = initial_rate
    data.ctrl[:] = 0.0

    mujoco.mj_forward(
        model,
        data,
    )

    rng = np.random.default_rng(
        seed
    )

    pos_motor_mask = mask_for(
        candidate["pos_motor"]
    )

    neg_motor_mask = mask_for(
        candidate["neg_motor"]
    )

    (
        pos_trace,
        neg_trace,
        baseline_ratio,
    ) = neutral_warmup(
        candidate,
        rng,
        pos_motor_mask,
        neg_motor_mask,
    )

    max_abs_gyro = abs(
        initial_rate
    )

    command_sum = 0.0
    command_abs_sum = 0.0
    n_commands = 0

    total_steps = int(
        round(
            CLOSED_LOOP_SECONDS
            / physics_dt
        )
    )

    for _ in range(total_steps):
        gyro = get_gyro(
            data
        )

        angular_rate = float(
            gyro[
                axis_index
            ]
        )

        max_abs_gyro = max(
            max_abs_gyro,
            abs(angular_rate),
        )

        differential = float(
            np.clip(
                GYRO_TO_HZ
                * angular_rate,
                -MAX_DIFFERENTIAL_HZ,
                MAX_DIFFERENTIAL_HZ,
            )
        )

        rate_pos = float(
            np.clip(
                BASE_RATE_HZ
                + differential,
                0.0,
                150.0,
            )
        )

        rate_neg = float(
            np.clip(
                BASE_RATE_HZ
                - differential,
                0.0,
                150.0,
            )
        )

        pos_spikes_total = 0
        neg_spikes_total = 0

        for _ in range(
            lif_steps_per_physics
        ):
            p_pos = (
                rate_pos
                * brain.dt_ms
                / 1000.0
            )

            p_neg = (
                rate_neg
                * brain.dt_ms
                / 1000.0
            )

            forced_pos = candidate["pos_sens"][
                rng.random(
                    len(candidate["pos_sens"])
                ) < p_pos
            ]

            forced_neg = candidate["neg_sens"][
                rng.random(
                    len(candidate["neg_sens"])
                ) < p_neg
            ]

            forced = np.concatenate([
                forced_pos,
                forced_neg,
            ])

            spikes, _ = brain.step(
                forced_spikes=forced
            )

            if len(spikes):
                pos_spikes_total += int(
                    pos_motor_mask[
                        spikes
                    ].sum()
                )

                neg_spikes_total += int(
                    neg_motor_mask[
                        spikes
                    ].sum()
                )

        pos_trace = (
            pos_trace
            * trace_decay
            + pos_spikes_total
        )

        neg_trace = (
            neg_trace
            * trace_decay
            + neg_spikes_total
        )

        raw_ratio = (
            pos_trace
            - neg_trace
        ) / (
            pos_trace
            + neg_trace
            + 1e-6
        )

        brain_signal = float(
            np.clip(
                decoder_sign
                * (
                    raw_ratio
                    - baseline_ratio
                ),
                -1.0,
                1.0,
            )
        )

        control = float(
            np.clip(
                MOTOR_GAIN
                * brain_signal,
                -1.0,
                1.0,
            )
        )

        data.ctrl[:] = 0.0
        data.ctrl[actuator_id] = control

        mujoco.mj_step(
            model,
            data,
        )

        command_sum += control
        command_abs_sum += abs(control)
        n_commands += 1

    final_gyro = float(
        get_gyro(
            data
        )[axis_index]
    )

    reduction_percent = (
        1.0
        - abs(final_gyro)
        / max(abs(initial_rate), 1e-9)
    ) * 100.0

    stable = bool(
        abs(final_gyro) < abs(initial_rate)
        and max_abs_gyro < 1.5 * abs(initial_rate)
    )

    return {
        "family": family,
        "candidate": candidate["name"],
        "initial_rate": initial_rate,
        "final_rate": final_gyro,
        "reduction_percent": reduction_percent,
        "max_abs_gyro": max_abs_gyro,
        "mean_control": (
            command_sum
            / max(n_commands, 1)
        ),
        "mean_abs_control": (
            command_abs_sum
            / max(n_commands, 1)
        ),
        "baseline_ratio": baseline_ratio,
        "stable": stable,
        "seed": seed,
    }


closed_loop_rows = []

for family in ("roll", "pitch", "yaw"):
    best = best_by_family.get(
        family
    )

    if best is None:
        print(
            family.upper(),
            ": no candidate"
        )
        continue

    if (
        best["correlation"]
        < MIN_CORRELATION_FOR_CLOSED_LOOP
    ):
        print(
            family.upper(),
            ": skipped closed-loop; "
            f"corr={best['correlation']:.3f}"
        )
        continue

    candidate = find_candidate(
        best["candidate"]
    )

    print()
    print(
        family.upper(),
        "using",
        candidate["name"],
    )

    for sign_index, initial_rate in enumerate(
        (
            +INITIAL_RATE_RAD_S,
            -INITIAL_RATE_RAD_S,
        )
    ):
        result = run_closed_loop(
            family=family,
            candidate=candidate,
            decoder_sign=float(
                best["decoder_sign"]
            ),
            initial_rate=initial_rate,
            seed=500000 + sign_index + 100 * (
                {"roll": 1, "pitch": 2, "yaw": 3}[family]
            ),
        )

        closed_loop_rows.append(
            result
        )

        print(
            f"  {initial_rate:+.2f} "
            f"-> {result['final_rate']:+.4f} "
            f"| reduction={result['reduction_percent']:.1f}% "
            f"| max={result['max_abs_gyro']:.3f} "
            f"| stable={result['stable']}"
        )

save_csv(
    CLOSED_LOOP_CSV,
    closed_loop_rows,
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL SUMMARY")

summary_lines = []

summary_lines.append(
    "NEUROFLIGHT FULL DIAGNOSTIC V3"
)

summary_lines.append(
    f"Timestamp: {timestamp}"
)

summary_lines.append("")
summary_lines.append(
    f"MaleCNS neurons: {brain.num_neurons}"
)

summary_lines.append(
    f"MaleCNS edges: {len(post_idx)}"
)

summary_lines.append("")

for family in (
    "roll",
    "pitch",
    "yaw",
):
    best = best_by_family.get(
        family
    )

    if best is None:
        summary_lines.append(
            f"{family.upper()}: no candidate"
        )

        continue

    summary_lines.append(
        f"{family.upper()} OPEN LOOP:"
    )

    summary_lines.append(
        f"  candidate: {best['candidate']}"
    )

    summary_lines.append(
        f"  corr: {best['correlation']:.3f}"
    )

    summary_lines.append(
        f"  endpoint SNR: {best['endpoint_snr']:.2f}"
    )

    summary_lines.append(
        f"  symmetry error: {best['symmetry_error']:.3f}"
    )

    summary_lines.append(
        f"  usable by strict screen: {best['usable']}"
    )

    family_closed = [
        row
        for row in closed_loop_rows
        if row["family"] == family
    ]

    if family_closed:
        summary_lines.append(
            f"{family.upper()} CLOSED LOOP:"
        )

        for row in family_closed:
            summary_lines.append(
                "  "
                f"{row['initial_rate']:+.2f} "
                f"-> {row['final_rate']:+.4f} "
                f"({row['reduction_percent']:.1f}% reduction), "
                f"stable={row['stable']}"
            )

    else:
        summary_lines.append(
            f"{family.upper()} CLOSED LOOP: not run"
        )

    summary_lines.append("")


# Overall computational conclusion.
def family_closed_ok(family):
    rows = [
        row
        for row in closed_loop_rows
        if row["family"] == family
    ]

    return bool(
        len(rows) == 2
        and all(
            row["stable"]
            and row["reduction_percent"] > 20.0
            for row in rows
        )
    )


roll_ok = family_closed_ok("roll")
pitch_ok = family_closed_ok("pitch")
yaw_ok = family_closed_ok("yaw")

summary_lines.append(
    "AUTOMATIC COMPUTATIONAL VERDICT:"
)

summary_lines.append(
    f"  roll isolated damping:  {'PASS' if roll_ok else 'NOT PASS'}"
)

summary_lines.append(
    f"  pitch isolated damping: {'PASS' if pitch_ok else 'NOT PASS'}"
)

summary_lines.append(
    f"  yaw isolated damping:   {'PASS' if yaw_ok else 'NOT PASS'}"
)

if roll_ok and pitch_ok and yaw_ok:
    summary_lines.append(
        "  next engineering step: combine the 3 axes in one MuJoCo controller."
    )
else:
    summary_lines.append(
        "  next engineering step: inspect only the family/families marked NOT PASS."
    )

summary_lines.append("")
summary_lines.append(
    "SCIENTIFIC LIMITATION:"
)

summary_lines.append(
    "  PASS here means the computational MaleCNS model provides a usable"
)

summary_lines.append(
    "  signed channel under our synthetic sensory encoding and can damp"
)

summary_lines.append(
    "  the corresponding MuJoCo angular velocity."
)

summary_lines.append(
    "  It does not by itself prove the same subtype-to-axis mapping in a real fly."
)

summary_text = "\n".join(
    summary_lines
)

print(
    summary_text
)

with open(
    SUMMARY_TXT,
    "w",
    encoding="utf-8",
) as file:
    file.write(
        summary_text
        + "\n"
    )


manifest = {
    "timestamp": timestamp,
    "output_dir": str(OUT_DIR),
    "files": {
        "console": str(CONSOLE_LOG),
        "direct_connectivity": str(DIRECT_CSV),
        "channel_responses": str(RESPONSES_CSV),
        "channel_summary": str(CHANNEL_SUMMARY_CSV),
        "closed_loop": str(CLOSED_LOOP_CSV),
        "summary": str(SUMMARY_TXT),
    },
    "population_report": population_report,
    "best_by_family": best_by_family,
    "automatic_verdict": {
        "roll": roll_ok,
        "pitch": pitch_ok,
        "yaw": yaw_ok,
    },
}

with open(
    MANIFEST_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        manifest,
        file,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 78)
print("FILES TO SEND BACK")
print("=" * 78)
print(SUMMARY_TXT)
print(CHANNEL_SUMMARY_CSV)
print(CLOSED_LOOP_CSV)
print(CONSOLE_LOG)
print()
print("Done.")
