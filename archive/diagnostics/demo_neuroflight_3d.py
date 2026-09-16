from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import math
import sys

import mujoco
import mujoco_menagerie as mm
import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


# ============================================================
# NEUROFLIGHT - DEMO 3D
#
# One MaleCNS instance controls roll + pitch + yaw simultaneously.
#
# Sensory encoder:
#   roll  -> all haltere L/R
#   pitch -> SNpp34+SNpp25 versus SNpp20
#   yaw   -> SApp L/R
#
# Shared sensory neurons are stimulated ONCE. Their contributions are
# summed into one firing rate before Poisson forcing.
#
# Decoder:
#   3 motor features are measured simultaneously and a 3x3 response
#   matrix is calibrated automatically. Its pseudo-inverse separates
#   roll/pitch/yaw readout and compensates first-order cross-talk.
#
# Under gravity this demo uses constant hover thrust only.
# MaleCNS controls the body moments, not altitude or XY position.
#
# IMPORTANT:
# Haltere-like gyro feedback damps angular velocity. It does not, by
# itself, provide an absolute "level" reference for a statically tilted
# drone. Start is level, with an angular-rate perturbation.
# ============================================================


CACHE = Path("data/male_cns/cache_v3")

BASE_RATE_HZ = 75.0
DELTA_RATE_HZ = 50.0
GYRO_TO_HZ = 30.0

TRACE_TAU_MS = 100.0

CAL_WARMUP_MS = 250.0
CAL_SETTLE_MS = 150.0
CAL_MEASURE_MS = 300.0
CAL_REPEATS = 2

MOTOR_GAIN = 0.80

SIM_SECONDS = 6.0
START_Z = 1.50

# Real 3-axis perturbation.
INITIAL_BODY_RATES = np.array(
    [0.80, -0.60, 0.50],
    dtype=float,
)

SHOW_VIEWER = "--headless" not in sys.argv
VIEWER_SYNC_EVERY = 5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/demo_3d") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_CSV = OUT_DIR / "flight.csv"
DECODER_JSON = OUT_DIR / "decoder.json"
SUMMARY_TXT = OUT_DIR / "SUMMARY.txt"


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


def save_csv(path, rows):
    if not rows:
        return

    fields = []
    seen = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def quat_to_euler_wxyz(q):
    """Returns roll, pitch, yaw in radians from MuJoCo w,x,y,z quaternion."""
    w, x, y, z = [float(v) for v in q]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=float)


def get_named_sensor(model, data, name):
    sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        name,
    )

    if sensor_id < 0:
        raise RuntimeError(f"Sensor not found: {name}")

    adr = int(model.sensor_adr[sensor_id])
    dim = int(model.sensor_dim[sensor_id])

    return np.asarray(
        data.sensordata[adr:adr + dim],
        dtype=float,
    ).copy()


class NeuroFlight3AxisController:
    def __init__(
        self,
        cache_dir=CACHE,
        seed=12345,
        verbose=True,
    ):
        self.cache_dir = Path(cache_dir)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        if self.verbose:
            print("Loading MaleCNS LIF...")

        self.brain = MaleCNSLIF(
            cache_dir=self.cache_dir,
            dt_ms=0.5,
            global_gain=0.65,
        )

        self.metadata = pd.read_feather(
            self.cache_dir / "neurons.feather"
        )

        self.body_ids = np.load(
            self.cache_dir / "body_ids.npy",
            mmap_mode="r",
        )

        body_to_cache = {
            int(body_id): index
            for index, body_id in enumerate(self.body_ids)
        }

        self.metadata["_cache_index"] = (
            self.metadata["bodyId"]
            .map(body_to_cache)
        )

        if self.metadata["_cache_index"].isna().any():
            raise RuntimeError(
                "metadata/body_ids alignment failed"
            )

        self.metadata["_cache_index"] = (
            self.metadata["_cache_index"]
            .astype(np.int64)
        )

        self.metadata["_side"] = self.metadata.apply(
            infer_side,
            axis=1,
        )

        self.metadata["_type_clean"] = (
            self.metadata["type"]
            .map(norm_type)
        )

        self._build_populations()
        self._build_encoder()
        self._build_motor_masks()

        self.trace_decay = math.exp(
            -self.brain.dt_ms / TRACE_TAU_MS
        )

        self.traces = np.zeros(4, dtype=float)
        self.runtime_baseline = np.zeros(3, dtype=float)

        self.decoder_baseline = None
        self.response_matrix = None
        self.decoder_matrix = None
        self.decoder_condition = None

        self.calibrate()

    def _by_types(self, types, side=None):
        wanted = {
            norm_type(x)
            for x in types
        }

        mask = self.metadata["_type_clean"].isin(
            wanted
        )

        if side is not None:
            mask &= self.metadata["_side"].eq(
                side
            )

        return self.metadata.loc[
            mask,
            "_cache_index"
        ].to_numpy(dtype=np.int64)

    def _by_subclass(self, subclass, side=None):
        values = (
            self.metadata["subclass"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = values.eq(
            str(subclass).strip().lower()
        )

        if side is not None:
            mask &= self.metadata["_side"].eq(
                side
            )

        return self.metadata.loc[
            mask,
            "_cache_index"
        ].to_numpy(dtype=np.int64)

    def _make_mask(self, indices):
        mask = np.zeros(
            self.brain.num_neurons,
            dtype=bool,
        )

        if len(indices):
            mask[indices] = True

        return mask

    def _build_populations(self):
        self.haltere_L = self._by_subclass(
            "haltere",
            "L",
        )

        self.haltere_R = self._by_subclass(
            "haltere",
            "R",
        )

        self.sapp_L = self._by_types(
            ["SApp"],
            "L",
        )

        self.sapp_R = self._by_types(
            ["SApp"],
            "R",
        )

        sn34_L = self._by_types(
            ["SNpp34"],
            "L",
        )

        sn34_R = self._by_types(
            ["SNpp34"],
            "R",
        )

        sn25_L = self._by_types(
            ["SNpp25"],
            "L",
        )

        sn25_R = self._by_types(
            ["SNpp25"],
            "R",
        )

        sn20_L = self._by_types(
            ["SNpp20"],
            "L",
        )

        sn20_R = self._by_types(
            ["SNpp20"],
            "R",
        )

        self.pitch_positive = np.concatenate([
            sn34_L,
            sn34_R,
            sn25_L,
            sn25_R,
        ])

        self.pitch_negative = np.concatenate([
            sn20_L,
            sn20_R,
        ])

        self.b1_L = self._by_types(
            ["b1 MN"],
            "L",
        )

        self.b1_R = self._by_types(
            ["b1 MN"],
            "R",
        )

        self.b2_L = self._by_types(
            ["b2 MN"],
            "L",
        )

        self.b2_R = self._by_types(
            ["b2 MN"],
            "R",
        )

        self.b3_L = self._by_types(
            ["b3 MN"],
            "L",
        )

        self.b3_R = self._by_types(
            ["b3 MN"],
            "R",
        )

        self.i1_L = self._by_types(
            ["i1 MN"],
            "L",
        )

        self.i1_R = self._by_types(
            ["i1 MN"],
            "R",
        )

        self.b12_L = np.concatenate([
            self.b1_L,
            self.b2_L,
        ])

        self.b12_R = np.concatenate([
            self.b1_R,
            self.b2_R,
        ])

        self.b3i1_L = np.concatenate([
            self.b3_L,
            self.i1_L,
        ])

        self.b3i1_R = np.concatenate([
            self.b3_R,
            self.i1_R,
        ])

        if self.verbose:
            print(
                "Haltere L/R:",
                len(self.haltere_L),
                "/",
                len(self.haltere_R),
            )

            print(
                "Pitch + / -:",
                len(self.pitch_positive),
                "/",
                len(self.pitch_negative),
            )

            print(
                "Yaw SApp L/R:",
                len(self.sapp_L),
                "/",
                len(self.sapp_R),
            )

    def _build_encoder(self):
        # The full haltere population is the union and receives baseline.
        union = sorted(
            set(
                int(x)
                for x in np.concatenate([
                    self.haltere_L,
                    self.haltere_R,
                ])
            )
        )

        self.sensory_indices = np.asarray(
            union,
            dtype=np.int64,
        )

        local = {
            int(neuron): i
            for i, neuron in enumerate(self.sensory_indices)
        }

        n = len(self.sensory_indices)

        # Rows: sensory neurons.
        # Columns: roll, pitch, yaw.
        self.encoder_coeff = np.zeros(
            (n, 3),
            dtype=float,
        )

        def add(indices, axis, value):
            for neuron in indices:
                neuron = int(neuron)

                if neuron not in local:
                    raise RuntimeError(
                        f"Sensory neuron {neuron} not in haltere union"
                    )

                self.encoder_coeff[
                    local[neuron],
                    axis,
                ] += float(value)

        # Roll: full left/right haltere differential.
        add(self.haltere_L, 0, +1.0)
        add(self.haltere_R, 0, -1.0)

        # Pitch: subtype contrast, bilateral.
        add(self.pitch_positive, 1, +1.0)
        add(self.pitch_negative, 1, -1.0)

        # Yaw: SApp left/right differential.
        add(self.sapp_L, 2, +1.0)
        add(self.sapp_R, 2, -1.0)

        self.base_rates = np.full(
            n,
            BASE_RATE_HZ,
            dtype=float,
        )

    def _build_motor_masks(self):
        self.mask_b12_L = self._make_mask(
            self.b12_L
        )

        self.mask_b12_R = self._make_mask(
            self.b12_R
        )

        self.mask_b3i1_L = self._make_mask(
            self.b3i1_L
        )

        self.mask_b3i1_R = self._make_mask(
            self.b3i1_R
        )

    def _feature_vector(self):
        b12_L, b12_R, b3i1_L, b3i1_R = self.traces

        return np.asarray(
            [
                b12_L - b12_R,
                (b12_L + b12_R) - (b3i1_L + b3i1_R),
                b3i1_L - b3i1_R,
            ],
            dtype=float,
        )

    def _sensory_rates(self, command_xyz):
        command_xyz = np.asarray(
            command_xyz,
            dtype=float,
        )

        rates = (
            self.base_rates
            + DELTA_RATE_HZ
            * (
                self.encoder_coeff
                @ command_xyz
            )
        )

        return np.clip(
            rates,
            0.0,
            150.0,
        )

    def _tick(self, rng, command_xyz):
        rates = self._sensory_rates(
            command_xyz
        )

        probability = np.clip(
            rates
            * self.brain.dt_ms
            / 1000.0,
            0.0,
            1.0,
        )

        forced = self.sensory_indices[
            rng.random(
                len(self.sensory_indices)
            ) < probability
        ]

        spikes, _ = self.brain.step(
            forced_spikes=forced
        )

        counts = np.zeros(
            4,
            dtype=float,
        )

        if len(spikes):
            counts[0] = self.mask_b12_L[
                spikes
            ].sum()

            counts[1] = self.mask_b12_R[
                spikes
            ].sum()

            counts[2] = self.mask_b3i1_L[
                spikes
            ].sum()

            counts[3] = self.mask_b3i1_R[
                spikes
            ].sum()

        self.traces = (
            self.traces
            * self.trace_decay
            + counts
        )

        return self._feature_vector()

    def _reset_brain_state(self):
        self.brain.reset()
        self.traces[:] = 0.0

    def _run_for_ms(
        self,
        rng,
        command_xyz,
        duration_ms,
        collect=False,
    ):
        steps = int(
            round(
                duration_ms
                / self.brain.dt_ms
            )
        )

        samples = []

        for _ in range(steps):
            feature = self._tick(
                rng,
                command_xyz,
            )

            if collect:
                samples.append(
                    feature.copy()
                )

        if not samples:
            return self._feature_vector()

        return np.mean(
            np.asarray(samples),
            axis=0,
        )

    def _measure_command(
        self,
        command_xyz,
        seed,
    ):
        self._reset_brain_state()
        rng = np.random.default_rng(
            seed
        )

        self._run_for_ms(
            rng,
            np.zeros(3),
            CAL_WARMUP_MS,
            collect=False,
        )

        self._run_for_ms(
            rng,
            command_xyz,
            CAL_SETTLE_MS,
            collect=False,
        )

        return self._run_for_ms(
            rng,
            command_xyz,
            CAL_MEASURE_MS,
            collect=True,
        )

    def calibrate(self):
        if self.verbose:
            print()
            print("Calibrating combined 3-axis decoder...")

        neutral_values = []

        for repeat in range(CAL_REPEATS):
            neutral_values.append(
                self._measure_command(
                    np.zeros(3),
                    seed=900000 + repeat,
                )
            )

        baseline = np.mean(
            np.asarray(neutral_values),
            axis=0,
        )

        response = np.zeros(
            (3, 3),
            dtype=float,
        )

        for axis in range(3):
            plus_values = []
            minus_values = []

            for repeat in range(CAL_REPEATS):
                plus = np.zeros(3)
                plus[axis] = +1.0

                minus = np.zeros(3)
                minus[axis] = -1.0

                plus_values.append(
                    self._measure_command(
                        plus,
                        seed=910000 + axis * 1000 + repeat,
                    )
                )

                minus_values.append(
                    self._measure_command(
                        minus,
                        seed=920000 + axis * 1000 + repeat,
                    )
                )

            plus_mean = np.mean(
                np.asarray(plus_values),
                axis=0,
            )

            minus_mean = np.mean(
                np.asarray(minus_values),
                axis=0,
            )

            response[:, axis] = (
                plus_mean
                - minus_mean
            ) / 2.0

        condition = float(
            np.linalg.cond(
                response
            )
        )

        decoder = np.linalg.pinv(
            response,
            rcond=1e-3,
        )

        self.decoder_baseline = baseline
        self.response_matrix = response
        self.decoder_matrix = decoder
        self.decoder_condition = condition

        if self.verbose:
            print("Neutral motor features:")
            print(
                np.array2string(
                    baseline,
                    precision=4,
                    suppress_small=True,
                )
            )

            print("3x3 response matrix:")
            print(
                np.array2string(
                    response,
                    precision=4,
                    suppress_small=True,
                )
            )

            print(
                "Condition number:",
                f"{condition:.3f}",
            )

            if not np.isfinite(condition) or condition > 100.0:
                print(
                    "WARNING: decoder matrix is poorly conditioned."
                )

    def reset_runtime(self, seed=None):
        if seed is None:
            seed = self.seed

        self._reset_brain_state()

        self.runtime_rng = np.random.default_rng(
            int(seed)
        )

        # Bring recurrent state and motor traces to neutral activity.
        self._run_for_ms(
            self.runtime_rng,
            np.zeros(3),
            CAL_WARMUP_MS,
            collect=False,
        )

        # Session-specific center reduces stochastic baseline mismatch.
        neutral_samples = []

        neutral_steps = int(
            round(
                100.0
                / self.brain.dt_ms
            )
        )

        for _ in range(neutral_steps):
            neutral_samples.append(
                self._tick(
                    self.runtime_rng,
                    np.zeros(3),
                ).copy()
            )

        self.runtime_baseline = np.mean(
            np.asarray(neutral_samples),
            axis=0,
        )

    def step_from_gyro(
        self,
        gyro_xyz,
        lif_steps,
    ):
        gyro_xyz = np.asarray(
            gyro_xyz,
            dtype=float,
        )

        # Same scale as the isolated tests:
        # 30 Hz per rad/s, clipped to +/-50 Hz.
        command = np.clip(
            (
                GYRO_TO_HZ
                * gyro_xyz
            )
            / DELTA_RATE_HZ,
            -1.0,
            1.0,
        )

        feature = self._feature_vector()

        for _ in range(int(lif_steps)):
            feature = self._tick(
                self.runtime_rng,
                command,
            )

        centered = (
            feature
            - self.runtime_baseline
        )

        decoded = (
            self.decoder_matrix
            @ centered
        )

        decoded = np.clip(
            decoded,
            -1.0,
            1.0,
        )

        return {
            "requested": command.copy(),
            "decoded": decoded.copy(),
            "features": feature.copy(),
            "rates_min": float(
                self._sensory_rates(command).min()
            ),
            "rates_max": float(
                self._sensory_rates(command).max()
            ),
        }


def get_actuator_id(model, name):
    actuator_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        name,
    )

    if actuator_id < 0:
        raise RuntimeError(
            f"Actuator not found: {name}"
        )

    return actuator_id


def clipped_ctrl(model, actuator_id, value):
    if bool(model.actuator_ctrllimited[actuator_id]):
        low, high = model.actuator_ctrlrange[
            actuator_id
        ]

        return float(
            np.clip(
                value,
                low,
                high,
            )
        )

    return float(value)


def run_demo():
    print()
    print("=" * 78)
    print("NEUROFLIGHT 3D DEMO")
    print("=" * 78)

    controller = NeuroFlight3AxisController(
        seed=4242,
        verbose=True,
    )

    model = mm.load(
        "bitcraze_crazyflie_2"
    )

    data = mujoco.MjData(
        model
    )

    thrust_id = get_actuator_id(
        model,
        "body_thrust",
    )

    moment_ids = [
        get_actuator_id(
            model,
            "x_moment",
        ),
        get_actuator_id(
            model,
            "y_moment",
        ),
        get_actuator_id(
            model,
            "z_moment",
        ),
    ]

    # Determine the control sign needed for negative feedback from
    # the rotational gear entries. Crazyflie currently uses negative
    # moment gears, therefore decoded + -> ctrl +.
    ctrl_sign = np.ones(
        3,
        dtype=float,
    )

    for axis, actuator_id in enumerate(moment_ids):
        gear = float(
            model.actuator_gear[
                actuator_id,
                3 + axis,
            ]
        )

        if abs(gear) > 1e-15:
            ctrl_sign[axis] = -np.sign(
                gear
            )

    mass = float(
        np.sum(
            model.body_mass
        )
    )

    gravity = abs(
        float(
            model.opt.gravity[2]
        )
    )

    thrust_gear_z = float(
        model.actuator_gear[
            thrust_id,
            2,
        ]
    )

    if abs(thrust_gear_z) < 1e-12:
        thrust_gear_z = 1.0

    hover_ctrl = (
        mass
        * gravity
        / abs(thrust_gear_z)
    )

    hover_ctrl = clipped_ctrl(
        model,
        thrust_id,
        hover_ctrl,
    )

    physics_dt = float(
        model.opt.timestep
    )

    lif_steps = max(
        1,
        int(
            round(
                physics_dt
                * 1000.0
                / controller.brain.dt_ms
            )
        ),
    )

    mujoco.mj_resetData(
        model,
        data,
    )

    data.qpos[0:3] = [
        0.0,
        0.0,
        START_Z,
    ]

    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    data.qvel[:] = 0.0
    data.qvel[3:6] = INITIAL_BODY_RATES

    data.ctrl[:] = 0.0
    data.ctrl[thrust_id] = hover_ctrl

    mujoco.mj_forward(
        model,
        data,
    )

    controller.reset_runtime(
        seed=777,
    )

    print()
    print(
        "Mass:",
        f"{mass:.6f} kg",
    )

    print(
        "Gravity:",
        f"{gravity:.4f} m/s^2",
    )

    print(
        "Hover thrust ctrl:",
        f"{hover_ctrl:.6f}",
    )

    print(
        "MuJoCo dt:",
        f"{physics_dt * 1000.0:.3f} ms",
    )

    print(
        "LIF steps / physics:",
        lif_steps,
    )

    print(
        "Initial body rates:",
        INITIAL_BODY_RATES,
    )

    print(
        "Viewer:",
        "ON" if SHOW_VIEWER else "HEADLESS",
    )

    decoder_payload = {
        "baseline_calibration": (
            controller.decoder_baseline
            .tolist()
        ),
        "runtime_baseline": (
            controller.runtime_baseline
            .tolist()
        ),
        "response_matrix": (
            controller.response_matrix
            .tolist()
        ),
        "decoder_matrix": (
            controller.decoder_matrix
            .tolist()
        ),
        "condition_number": (
            controller.decoder_condition
        ),
        "ctrl_sign": (
            ctrl_sign.tolist()
        ),
        "hover_ctrl": hover_ctrl,
    }

    with open(
        DECODER_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            decoder_payload,
            file,
            indent=2,
        )

    total_steps = int(
        round(
            SIM_SECONDS
            / physics_dt
        )
    )

    log_every = max(
        1,
        int(
            round(
                0.05
                / physics_dt
            )
        ),
    )

    rows = []
    initial_gyro = None
    max_gyro_norm = 0.0
    min_z = float("inf")

    def simulation_loop(viewer=None):
        nonlocal initial_gyro
        nonlocal max_gyro_norm
        nonlocal min_z

        for step in range(total_steps):
            gyro = get_named_sensor(
                model,
                data,
                "body_gyro",
            )

            if initial_gyro is None:
                initial_gyro = gyro.copy()

            brain_out = controller.step_from_gyro(
                gyro,
                lif_steps,
            )

            moment_ctrl = (
                ctrl_sign
                * MOTOR_GAIN
                * brain_out["decoded"]
            )

            data.ctrl[:] = 0.0
            data.ctrl[thrust_id] = hover_ctrl

            for axis in range(3):
                data.ctrl[
                    moment_ids[axis]
                ] = clipped_ctrl(
                    model,
                    moment_ids[axis],
                    moment_ctrl[axis],
                )

            mujoco.mj_step(
                model,
                data,
            )

            gyro_after = get_named_sensor(
                model,
                data,
                "body_gyro",
            )

            euler = quat_to_euler_wxyz(
                data.qpos[3:7]
            )

            gyro_norm = float(
                np.linalg.norm(
                    gyro_after
                )
            )

            max_gyro_norm = max(
                max_gyro_norm,
                gyro_norm,
            )

            min_z = min(
                min_z,
                float(data.qpos[2]),
            )

            if step % log_every == 0:
                rows.append({
                    "time": float(data.time),
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "z": float(data.qpos[2]),
                    "roll_deg": math.degrees(euler[0]),
                    "pitch_deg": math.degrees(euler[1]),
                    "yaw_deg": math.degrees(euler[2]),
                    "gyro_x": float(gyro_after[0]),
                    "gyro_y": float(gyro_after[1]),
                    "gyro_z": float(gyro_after[2]),
                    "gyro_norm": gyro_norm,
                    "requested_roll": float(brain_out["requested"][0]),
                    "requested_pitch": float(brain_out["requested"][1]),
                    "requested_yaw": float(brain_out["requested"][2]),
                    "decoded_roll": float(brain_out["decoded"][0]),
                    "decoded_pitch": float(brain_out["decoded"][1]),
                    "decoded_yaw": float(brain_out["decoded"][2]),
                    "ctrl_x": float(data.ctrl[moment_ids[0]]),
                    "ctrl_y": float(data.ctrl[moment_ids[1]]),
                    "ctrl_z": float(data.ctrl[moment_ids[2]]),
                    "thrust": float(data.ctrl[thrust_id]),
                    "sensory_rate_min": brain_out["rates_min"],
                    "sensory_rate_max": brain_out["rates_max"],
                })

            if viewer is not None and step % VIEWER_SYNC_EVERY == 0:
                viewer.sync()

            # Stop if the model has fallen substantially through/onto floor.
            if float(data.qpos[2]) < -0.20:
                print(
                    "Emergency stop: z < -0.20 m"
                )
                break

    if SHOW_VIEWER:
        try:
            from mujoco import viewer as mj_viewer

            viewer = mj_viewer.launch_passive(
                model,
                data,
            )

            try:
                simulation_loop(
                    viewer
                )
            finally:
                viewer.close()

        except Exception as exc:
            print(
                "Viewer unavailable, continuing headless:",
                repr(exc),
            )

            simulation_loop(
                None
            )
    else:
        simulation_loop(
            None
        )

    save_csv(
        FLIGHT_CSV,
        rows,
    )

    final_gyro = get_named_sensor(
        model,
        data,
        "body_gyro",
    )

    initial_norm = float(
        np.linalg.norm(
            initial_gyro
        )
    )

    final_norm = float(
        np.linalg.norm(
            final_gyro
        )
    )

    reduction = (
        1.0
        - final_norm
        / max(initial_norm, 1e-9)
    ) * 100.0

    if rows:
        last_rows = [
            row
            for row in rows
            if row["time"] >= max(
                0.0,
                rows[-1]["time"] - 1.0,
            )
        ]

        last_mean_gyro = float(
            np.mean(
                [
                    row["gyro_norm"]
                    for row in last_rows
                ]
            )
        )
    else:
        last_mean_gyro = float("nan")

    summary = (
        "NEUROFLIGHT 3D DEMO\n"
        f"Initial gyro norm: {initial_norm:.4f} rad/s\n"
        f"Final gyro norm:   {final_norm:.4f} rad/s\n"
        f"Reduction:         {reduction:.1f}%\n"
        f"Last-1s mean gyro: {last_mean_gyro:.4f} rad/s\n"
        f"Maximum gyro norm: {max_gyro_norm:.4f} rad/s\n"
        f"Final position:    x={data.qpos[0]:+.3f}, "
        f"y={data.qpos[1]:+.3f}, z={data.qpos[2]:+.3f} m\n"
        f"Minimum z:         {min_z:.3f} m\n"
        f"Decoder condition: {controller.decoder_condition:.3f}\n"
        "\n"
        "Interpretation:\n"
        "- MaleCNS controls roll/pitch/yaw body moments simultaneously.\n"
        "- Thrust is fixed at physical hover thrust.\n"
        "- No XY position controller and no absolute attitude reference are used.\n"
        "- Therefore some translation/altitude loss after a large rotational "
        "perturbation is possible even when angular-rate damping works.\n"
    )

    with open(
        SUMMARY_TXT,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            summary
        )

    print()
    print("=" * 78)
    print(summary)
    print("Files:")
    print(SUMMARY_TXT)
    print(FLIGHT_CSV)
    print(DECODER_JSON)


if __name__ == "__main__":
    run_demo()
