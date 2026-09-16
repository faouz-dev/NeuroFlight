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
# NEUROFLIGHT - 3D ATTITUDE DEMO V2
#
# What changed after the first real 3D test:
#   - the old instantaneous pseudo-inverse decoder saturated
#     pitch/yaw and was noisy on roll;
#   - haltere/gyro feedback alone damps angular velocity but cannot
#     know that a statically tilted vehicle should return to level.
#
# V2 therefore:
#   1) trains one regularized 3-axis neural readout on BOTH isolated
#      and simultaneous sensory commands;
#   2) aligns the runtime neutral baseline to the training baseline;
#   3) low-pass filters the decoded neural command;
#   4) adds an explicit OUTER attitude reference:
#          effective_rate = gyro + K_attitude * attitude
#      The MaleCNS computational model remains the source of the
#      roll/pitch/yaw moment command.
#   5) uses a classical altitude-hold/thrust adapter so a successful
#      attitude test does not simply fall because the thrust vector
#      was temporarily tilted.
#
# Scientific attribution:
#   - body moments: MaleCNS computational controller/readout;
#   - absolute level reference: engineering attitude adapter;
#   - altitude/thrust: classical engineering adapter.
# ============================================================


CACHE = Path("data/male_cns/cache_v3")

BASE_RATE_HZ = 75.0
DELTA_RATE_HZ = 50.0
GYRO_TO_HZ = 30.0

TRACE_TAU_MS = 100.0
DECODE_TAU_MS = 45.0

CAL_WARMUP_MS = 220.0
CAL_SETTLE_MS = 150.0
CAL_MEASURE_MS = 300.0
CAL_SAMPLE_EVERY_MS = 5.0
RIDGE_LAMBDA = 1.0

MOTOR_GAIN = 0.80

# Outer engineering attitude reference.
ATTITUDE_K = np.array(
    [1.35, 1.35, 0.85],
    dtype=float,
)

# Classical vertical hold.
KP_Z = 0.16
KD_Z = 0.085

SIM_SECONDS = 8.0
START_Z = 1.50

INITIAL_EULER_DEG = np.array(
    [12.0, -8.0, 10.0],
    dtype=float,
)

INITIAL_BODY_RATES = np.array(
    [0.30, -0.25, 0.20],
    dtype=float,
)

SHOW_VIEWER = "--headless" not in sys.argv
VIEWER_SYNC_EVERY = 5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/demo_3d_v2") / timestamp
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

    return np.array(
        [roll, pitch, yaw],
        dtype=float,
    )


def euler_to_quat_wxyz(roll, pitch, yaw):
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)


def get_named_sensor(model, data, name):
    sensor_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SENSOR,
        name,
    )

    if sensor_id < 0:
        raise RuntimeError(
            f"Sensor not found: {name}"
        )

    adr = int(
        model.sensor_adr[sensor_id]
    )

    dim = int(
        model.sensor_dim[sensor_id]
    )

    return np.asarray(
        data.sensordata[
            adr:adr + dim
        ],
        dtype=float,
    ).copy()


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
    if bool(
        model.actuator_ctrllimited[
            actuator_id
        ]
    ):
        low, high = (
            model.actuator_ctrlrange[
                actuator_id
            ]
        )

        return float(
            np.clip(
                value,
                low,
                high,
            )
        )

    return float(value)


class NeuroFlight3AxisControllerV2:
    def __init__(
        self,
        cache_dir=CACHE,
        seed=12345,
        verbose=True,
    ):
        self.cache_dir = Path(
            cache_dir
        )

        self.seed = int(seed)
        self.verbose = bool(verbose)

        if self.verbose:
            print(
                "Loading MaleCNS LIF..."
            )

        self.brain = MaleCNSLIF(
            cache_dir=self.cache_dir,
            dt_ms=0.5,
            global_gain=0.65,
        )

        self.metadata = pd.read_feather(
            self.cache_dir
            / "neurons.feather"
        )

        self.body_ids = np.load(
            self.cache_dir
            / "body_ids.npy",
            mmap_mode="r",
        )

        body_to_cache = {
            int(body_id): index
            for index, body_id
            in enumerate(self.body_ids)
        }

        self.metadata[
            "_cache_index"
        ] = (
            self.metadata["bodyId"]
            .map(body_to_cache)
        )

        if self.metadata[
            "_cache_index"
        ].isna().any():
            raise RuntimeError(
                "metadata/body_ids alignment failed"
            )

        self.metadata[
            "_cache_index"
        ] = (
            self.metadata[
                "_cache_index"
            ]
            .astype(np.int64)
        )

        self.metadata[
            "_side"
        ] = self.metadata.apply(
            infer_side,
            axis=1,
        )

        self.metadata[
            "_type_clean"
        ] = (
            self.metadata["type"]
            .map(norm_type)
        )

        self._build_populations()
        self._build_encoder()
        self._build_motor_masks()

        self.trace_decay = math.exp(
            -self.brain.dt_ms
            / TRACE_TAU_MS
        )

        self.decode_decay = math.exp(
            -self.brain.dt_ms
            / DECODE_TAU_MS
        )

        self.traces = np.zeros(
            4,
            dtype=float,
        )

        self.decoded_filter = np.zeros(
            3,
            dtype=float,
        )

        self.training_neutral = np.zeros(
            3,
            dtype=float,
        )

        self.runtime_neutral = np.zeros(
            3,
            dtype=float,
        )

        self.feature_mean = None
        self.feature_std = None
        self.regression = None
        self.training_rmse = None

        self.calibrate_decoder()

    def _by_types(
        self,
        types,
        side=None,
    ):
        wanted = {
            norm_type(x)
            for x in types
        }

        mask = (
            self.metadata[
                "_type_clean"
            ]
            .isin(wanted)
        )

        if side is not None:
            mask &= (
                self.metadata[
                    "_side"
                ]
                .eq(side)
            )

        return self.metadata.loc[
            mask,
            "_cache_index",
        ].to_numpy(
            dtype=np.int64
        )

    def _by_subclass(
        self,
        subclass,
        side=None,
    ):
        values = (
            self.metadata[
                "subclass"
            ]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
        )

        mask = values.eq(
            str(subclass)
            .strip()
            .lower()
        )

        if side is not None:
            mask &= (
                self.metadata[
                    "_side"
                ]
                .eq(side)
            )

        return self.metadata.loc[
            mask,
            "_cache_index",
        ].to_numpy(
            dtype=np.int64
        )

    def _make_mask(
        self,
        indices,
    ):
        mask = np.zeros(
            self.brain.num_neurons,
            dtype=bool,
        )

        if len(indices):
            mask[
                indices
            ] = True

        return mask

    def _build_populations(self):
        self.haltere_L = (
            self._by_subclass(
                "haltere",
                "L",
            )
        )

        self.haltere_R = (
            self._by_subclass(
                "haltere",
                "R",
            )
        )

        self.sapp_L = (
            self._by_types(
                ["SApp"],
                "L",
            )
        )

        self.sapp_R = (
            self._by_types(
                ["SApp"],
                "R",
            )
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

        self.pitch_positive = (
            np.concatenate([
                sn34_L,
                sn34_R,
                sn25_L,
                sn25_R,
            ])
        )

        self.pitch_negative = (
            np.concatenate([
                sn20_L,
                sn20_R,
            ])
        )

        b1_L = self._by_types(
            ["b1 MN"],
            "L",
        )

        b1_R = self._by_types(
            ["b1 MN"],
            "R",
        )

        b2_L = self._by_types(
            ["b2 MN"],
            "L",
        )

        b2_R = self._by_types(
            ["b2 MN"],
            "R",
        )

        b3_L = self._by_types(
            ["b3 MN"],
            "L",
        )

        b3_R = self._by_types(
            ["b3 MN"],
            "R",
        )

        i1_L = self._by_types(
            ["i1 MN"],
            "L",
        )

        i1_R = self._by_types(
            ["i1 MN"],
            "R",
        )

        self.b12_L = (
            np.concatenate([
                b1_L,
                b2_L,
            ])
        )

        self.b12_R = (
            np.concatenate([
                b1_R,
                b2_R,
            ])
        )

        self.b3i1_L = (
            np.concatenate([
                b3_L,
                i1_L,
            ])
        )

        self.b3i1_R = (
            np.concatenate([
                b3_R,
                i1_R,
            ])
        )

        if self.verbose:
            print(
                "Haltere L/R:",
                len(self.haltere_L),
                "/",
                len(self.haltere_R),
            )

            print(
                "Pitch subtype +/-:",
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
        sensory_union = sorted(
            set(
                int(x)
                for x in np.concatenate([
                    self.haltere_L,
                    self.haltere_R,
                ])
            )
        )

        self.sensory_indices = (
            np.asarray(
                sensory_union,
                dtype=np.int64,
            )
        )

        local = {
            int(neuron): index
            for index, neuron
            in enumerate(
                self.sensory_indices
            )
        }

        self.encoder_coeff = (
            np.zeros(
                (
                    len(
                        self.sensory_indices
                    ),
                    3,
                ),
                dtype=float,
            )
        )

        def add(
            indices,
            axis,
            value,
        ):
            for neuron in indices:
                neuron = int(
                    neuron
                )

                self.encoder_coeff[
                    local[neuron],
                    axis,
                ] += float(
                    value
                )

        # Roll: all haltere L/R.
        add(
            self.haltere_L,
            0,
            +1.0,
        )

        add(
            self.haltere_R,
            0,
            -1.0,
        )

        # Pitch: subtype contrast.
        add(
            self.pitch_positive,
            1,
            +1.0,
        )

        add(
            self.pitch_negative,
            1,
            -1.0,
        )

        # Yaw: SApp L/R.
        add(
            self.sapp_L,
            2,
            +1.0,
        )

        add(
            self.sapp_R,
            2,
            -1.0,
        )

        self.base_rates = np.full(
            len(
                self.sensory_indices
            ),
            BASE_RATE_HZ,
            dtype=float,
        )

    def _build_motor_masks(self):
        self.mask_b12_L = (
            self._make_mask(
                self.b12_L
            )
        )

        self.mask_b12_R = (
            self._make_mask(
                self.b12_R
            )
        )

        self.mask_b3i1_L = (
            self._make_mask(
                self.b3i1_L
            )
        )

        self.mask_b3i1_R = (
            self._make_mask(
                self.b3i1_R
            )
        )

    def _reset_brain_state(self):
        self.brain.reset()

        self.traces[:] = 0.0

        self.decoded_filter[:] = 0.0

    def _feature_vector(self):
        (
            b12_L,
            b12_R,
            b3i1_L,
            b3i1_R,
        ) = self.traces

        return np.asarray(
            [
                b12_L
                - b12_R,

                (
                    b12_L
                    + b12_R
                )
                -
                (
                    b3i1_L
                    + b3i1_R
                ),

                b3i1_L
                - b3i1_R,
            ],
            dtype=float,
        )

    def _sensory_rates(
        self,
        command_xyz,
    ):
        command_xyz = (
            np.asarray(
                command_xyz,
                dtype=float,
            )
        )

        rates = (
            self.base_rates
            +
            DELTA_RATE_HZ
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

    def _tick(
        self,
        rng,
        command_xyz,
    ):
        rates = (
            self._sensory_rates(
                command_xyz
            )
        )

        probability = (
            np.clip(
                rates
                * self.brain.dt_ms
                / 1000.0,
                0.0,
                1.0,
            )
        )

        forced = (
            self.sensory_indices[
                rng.random(
                    len(
                        self.sensory_indices
                    )
                )
                < probability
            ]
        )

        spikes, _ = (
            self.brain.step(
                forced_spikes=forced
            )
        )

        counts = np.zeros(
            4,
            dtype=float,
        )

        if len(spikes):
            counts[0] = (
                self.mask_b12_L[
                    spikes
                ]
                .sum()
            )

            counts[1] = (
                self.mask_b12_R[
                    spikes
                ]
                .sum()
            )

            counts[2] = (
                self.mask_b3i1_L[
                    spikes
                ]
                .sum()
            )

            counts[3] = (
                self.mask_b3i1_R[
                    spikes
                ]
                .sum()
            )

        self.traces = (
            self.traces
            * self.trace_decay
            + counts
        )

        return (
            self._feature_vector()
        )

    def _training_points(self):
        points = [
            np.zeros(
                3,
                dtype=float,
            )
        ]

        for axis in range(3):
            for sign in (
                -1.0,
                +1.0,
            ):
                point = np.zeros(
                    3,
                    dtype=float,
                )

                point[
                    axis
                ] = sign

                points.append(
                    point
                )

        # Simultaneous commands teach the decoder the cross-talk
        # that the first 3D script did not see during calibration.
        for sx in (-0.55, +0.55):
            for sy in (-0.55, +0.55):
                for sz in (-0.55, +0.55):
                    points.append(
                        np.asarray(
                            [sx, sy, sz],
                            dtype=float,
                        )
                    )

        return points

    def _collect_training_samples(
        self,
        command,
        seed,
    ):
        self._reset_brain_state()

        rng = (
            np.random.default_rng(
                seed
            )
        )

        warm_steps = int(
            round(
                CAL_WARMUP_MS
                / self.brain.dt_ms
            )
        )

        for _ in range(
            warm_steps
        ):
            self._tick(
                rng,
                np.zeros(3),
            )

        settle_steps = int(
            round(
                CAL_SETTLE_MS
                / self.brain.dt_ms
            )
        )

        for _ in range(
            settle_steps
        ):
            self._tick(
                rng,
                command,
            )

        measure_steps = int(
            round(
                CAL_MEASURE_MS
                / self.brain.dt_ms
            )
        )

        sample_every = max(
            1,
            int(
                round(
                    CAL_SAMPLE_EVERY_MS
                    / self.brain.dt_ms
                )
            ),
        )

        samples = []

        for step in range(
            measure_steps
        ):
            feature = self._tick(
                rng,
                command,
            )

            if (
                step
                % sample_every
                == 0
            ):
                samples.append(
                    feature.copy()
                )

        return np.asarray(
            samples,
            dtype=float,
        )

    def calibrate_decoder(self):
        if self.verbose:
            print()
            print(
                "Training regularized combined decoder..."
            )

        points = (
            self._training_points()
        )

        all_features = []
        all_targets = []

        neutral_samples = None

        for index, command in enumerate(
            points
        ):
            features = (
                self._collect_training_samples(
                    command,
                    seed=(
                        900000
                        + index
                    ),
                )
            )

            if np.allclose(
                command,
                0.0,
            ):
                neutral_samples = (
                    features.copy()
                )

            all_features.append(
                features
            )

            all_targets.append(
                np.repeat(
                    command[
                        None,
                        :
                    ],
                    len(features),
                    axis=0,
                )
            )

        X = np.concatenate(
            all_features,
            axis=0,
        )

        Y = np.concatenate(
            all_targets,
            axis=0,
        )

        if neutral_samples is None:
            raise RuntimeError(
                "Neutral calibration samples missing"
            )

        self.training_neutral = (
            np.mean(
                neutral_samples,
                axis=0,
            )
        )

        self.feature_mean = np.mean(
            X,
            axis=0,
        )

        self.feature_std = np.std(
            X,
            axis=0,
        )

        self.feature_std = np.maximum(
            self.feature_std,
            1e-6,
        )

        Z = (
            X
            - self.feature_mean
        ) / self.feature_std

        A = np.column_stack([
            np.ones(
                len(Z)
            ),
            Z,
        ])

        regularizer = np.eye(
            A.shape[1],
            dtype=float,
        )

        # Do not penalize intercept.
        regularizer[
            0,
            0,
        ] = 0.0

        lhs = (
            A.T
            @ A
            +
            RIDGE_LAMBDA
            * regularizer
        )

        rhs = (
            A.T
            @ Y
        )

        self.regression = (
            np.linalg.solve(
                lhs,
                rhs,
            )
        )

        prediction = (
            A
            @ self.regression
        )

        self.training_rmse = np.sqrt(
            np.mean(
                (
                    prediction
                    - Y
                )
                ** 2,
                axis=0,
            )
        )

        if self.verbose:
            print(
                "Training samples:",
                len(X),
            )

            print(
                "Training neutral:",
                np.array2string(
                    self.training_neutral,
                    precision=3,
                ),
            )

            print(
                "Decoder RMSE roll/pitch/yaw:",
                np.array2string(
                    self.training_rmse,
                    precision=3,
                ),
            )

    def _decode_feature(
        self,
        feature,
    ):
        # Align session-specific neutral state with the neutral state
        # that the regression saw during training.
        aligned = (
            feature
            - self.runtime_neutral
            + self.training_neutral
        )

        z = (
            aligned
            - self.feature_mean
        ) / self.feature_std

        a = np.concatenate([
            [1.0],
            z,
        ])

        decoded_raw = (
            a
            @ self.regression
        )

        # Additional short low-pass on the neural readout.
        self.decoded_filter = (
            self.decoded_filter
            * self.decode_decay
            +
            decoded_raw
            * (
                1.0
                - self.decode_decay
            )
        )

        return np.clip(
            self.decoded_filter,
            -1.0,
            1.0,
        )

    def reset_runtime(
        self,
        seed=None,
    ):
        if seed is None:
            seed = self.seed

        self._reset_brain_state()

        self.runtime_rng = (
            np.random.default_rng(
                int(seed)
            )
        )

        warm_steps = int(
            round(
                250.0
                / self.brain.dt_ms
            )
        )

        for _ in range(
            warm_steps
        ):
            self._tick(
                self.runtime_rng,
                np.zeros(3),
            )

        neutral_features = []

        neutral_steps = int(
            round(
                150.0
                / self.brain.dt_ms
            )
        )

        for _ in range(
            neutral_steps
        ):
            neutral_features.append(
                self._tick(
                    self.runtime_rng,
                    np.zeros(3),
                ).copy()
            )

        self.runtime_neutral = (
            np.mean(
                np.asarray(
                    neutral_features
                ),
                axis=0,
            )
        )

        self.decoded_filter[:] = 0.0

        if self.verbose:
            print(
                "Runtime neutral:",
                np.array2string(
                    self.runtime_neutral,
                    precision=3,
                ),
            )

    def step_from_effective_rate(
        self,
        effective_rate_xyz,
        lif_steps,
    ):
        effective_rate_xyz = (
            np.asarray(
                effective_rate_xyz,
                dtype=float,
            )
        )

        requested = np.clip(
            (
                GYRO_TO_HZ
                * effective_rate_xyz
            )
            / DELTA_RATE_HZ,
            -1.0,
            1.0,
        )

        feature = (
            self._feature_vector()
        )

        for _ in range(
            int(lif_steps)
        ):
            feature = self._tick(
                self.runtime_rng,
                requested,
            )

        decoded = (
            self._decode_feature(
                feature
            )
        )

        rates = (
            self._sensory_rates(
                requested
            )
        )

        return {
            "requested": (
                requested.copy()
            ),
            "decoded": (
                decoded.copy()
            ),
            "features": (
                feature.copy()
            ),
            "rates_min": float(
                rates.min()
            ),
            "rates_max": float(
                rates.max()
            ),
        }


def build_vehicle():
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

    ctrl_sign = np.ones(
        3,
        dtype=float,
    )

    for axis, actuator_id in enumerate(
        moment_ids
    ):
        gear = float(
            model.actuator_gear[
                actuator_id,
                3 + axis,
            ]
        )

        if abs(
            gear
        ) > 1e-15:
            ctrl_sign[
                axis
            ] = -np.sign(
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

    if abs(
        thrust_gear_z
    ) < 1e-12:
        thrust_gear_z = 1.0

    hover_ctrl = (
        mass
        * gravity
        / abs(
            thrust_gear_z
        )
    )

    hover_ctrl = clipped_ctrl(
        model,
        thrust_id,
        hover_ctrl,
    )

    return (
        model,
        data,
        thrust_id,
        moment_ids,
        ctrl_sign,
        hover_ctrl,
    )


def attitude_effective_rate(
    gyro_xyz,
    euler_xyz,
):
    return (
        np.asarray(
            gyro_xyz,
            dtype=float,
        )
        +
        ATTITUDE_K
        * np.asarray(
            euler_xyz,
            dtype=float,
        )
    )


def altitude_thrust(
    model,
    thrust_id,
    hover_ctrl,
    z,
    vz,
    target_z,
    target_vz,
    roll,
    pitch,
):
    # Tilt compensation is an engineering adapter, not MaleCNS.
    vertical_fraction = (
        math.cos(
            float(roll)
        )
        * math.cos(
            float(pitch)
        )
    )

    vertical_fraction = max(
        0.55,
        vertical_fraction,
    )

    tilt_compensated_hover = (
        hover_ctrl
        / vertical_fraction
    )

    thrust = (
        tilt_compensated_hover
        +
        KP_Z
        * (
            target_z
            - z
        )
        +
        KD_Z
        * (
            target_vz
            - vz
        )
    )

    return clipped_ctrl(
        model,
        thrust_id,
        thrust,
    )


def run_demo():
    print()
    print("=" * 78)
    print(
        "NEUROFLIGHT 3D ATTITUDE DEMO V2"
    )
    print("=" * 78)

    controller = (
        NeuroFlight3AxisControllerV2(
            seed=4242,
            verbose=True,
        )
    )

    (
        model,
        data,
        thrust_id,
        moment_ids,
        ctrl_sign,
        hover_ctrl,
    ) = build_vehicle()

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

    data.qpos[
        0:3
    ] = [
        0.0,
        0.0,
        START_Z,
    ]

    initial_euler = (
        np.radians(
            INITIAL_EULER_DEG
        )
    )

    data.qpos[
        3:7
    ] = euler_to_quat_wxyz(
        initial_euler[0],
        initial_euler[1],
        initial_euler[2],
    )

    data.qvel[:] = 0.0

    data.qvel[
        3:6
    ] = (
        INITIAL_BODY_RATES
    )

    data.ctrl[:] = 0.0

    data.ctrl[
        thrust_id
    ] = hover_ctrl

    mujoco.mj_forward(
        model,
        data,
    )

    controller.reset_runtime(
        seed=777,
    )

    print()
    print(
        "Hover thrust:",
        f"{hover_ctrl:.6f}",
    )

    print(
        "Initial attitude deg:",
        INITIAL_EULER_DEG,
    )

    print(
        "Initial rates rad/s:",
        INITIAL_BODY_RATES,
    )

    print(
        "Viewer:",
        (
            "ON"
            if SHOW_VIEWER
            else "HEADLESS"
        ),
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

    max_tilt_deg = 0.0
    max_gyro = 0.0
    min_z = float(
        "inf"
    )

    touched_ground = False

    def simulation_loop(
        viewer=None,
    ):
        nonlocal max_tilt_deg
        nonlocal max_gyro
        nonlocal min_z
        nonlocal touched_ground

        for step in range(
            total_steps
        ):
            gyro_xyz = (
                get_named_sensor(
                    model,
                    data,
                    "body_gyro",
                )
            )

            euler_xyz = (
                quat_to_euler_wxyz(
                    data.qpos[
                        3:7
                    ]
                )
            )

            effective_rate = (
                attitude_effective_rate(
                    gyro_xyz,
                    euler_xyz,
                )
            )

            brain_out = (
                controller.step_from_effective_rate(
                    effective_rate,
                    lif_steps,
                )
            )

            moment_ctrl = (
                ctrl_sign
                * MOTOR_GAIN
                * brain_out[
                    "decoded"
                ]
            )

            z = float(
                data.qpos[2]
            )

            vz = float(
                data.qvel[2]
            )

            thrust_cmd = (
                altitude_thrust(
                    model,
                    thrust_id,
                    hover_ctrl,
                    z,
                    vz,
                    START_Z,
                    0.0,
                    euler_xyz[0],
                    euler_xyz[1],
                )
            )

            data.ctrl[:] = 0.0

            data.ctrl[
                thrust_id
            ] = thrust_cmd

            for axis in range(3):
                data.ctrl[
                    moment_ids[
                        axis
                    ]
                ] = clipped_ctrl(
                    model,
                    moment_ids[
                        axis
                    ],
                    moment_ctrl[
                        axis
                    ],
                )

            mujoco.mj_step(
                model,
                data,
            )

            gyro_after = (
                get_named_sensor(
                    model,
                    data,
                    "body_gyro",
                )
            )

            euler_after = (
                quat_to_euler_wxyz(
                    data.qpos[
                        3:7
                    ]
                )
            )

            tilt_deg = max(
                abs(
                    math.degrees(
                        euler_after[0]
                    )
                ),
                abs(
                    math.degrees(
                        euler_after[1]
                    )
                ),
            )

            gyro_norm = float(
                np.linalg.norm(
                    gyro_after
                )
            )

            max_tilt_deg = max(
                max_tilt_deg,
                tilt_deg,
            )

            max_gyro = max(
                max_gyro,
                gyro_norm,
            )

            min_z = min(
                min_z,
                float(
                    data.qpos[2]
                ),
            )

            if (
                data.ncon > 0
                and float(
                    data.qpos[2]
                ) < 0.15
            ):
                touched_ground = True

            if (
                step
                % log_every
                == 0
            ):
                rows.append({
                    "time": float(
                        data.time
                    ),
                    "x": float(
                        data.qpos[0]
                    ),
                    "y": float(
                        data.qpos[1]
                    ),
                    "z": float(
                        data.qpos[2]
                    ),
                    "vz": float(
                        data.qvel[2]
                    ),
                    "roll_deg": (
                        math.degrees(
                            euler_after[0]
                        )
                    ),
                    "pitch_deg": (
                        math.degrees(
                            euler_after[1]
                        )
                    ),
                    "yaw_deg": (
                        math.degrees(
                            euler_after[2]
                        )
                    ),
                    "gyro_x": float(
                        gyro_after[0]
                    ),
                    "gyro_y": float(
                        gyro_after[1]
                    ),
                    "gyro_z": float(
                        gyro_after[2]
                    ),
                    "gyro_norm": (
                        gyro_norm
                    ),
                    "effective_x": float(
                        effective_rate[0]
                    ),
                    "effective_y": float(
                        effective_rate[1]
                    ),
                    "effective_z": float(
                        effective_rate[2]
                    ),
                    "requested_roll": float(
                        brain_out[
                            "requested"
                        ][0]
                    ),
                    "requested_pitch": float(
                        brain_out[
                            "requested"
                        ][1]
                    ),
                    "requested_yaw": float(
                        brain_out[
                            "requested"
                        ][2]
                    ),
                    "decoded_roll": float(
                        brain_out[
                            "decoded"
                        ][0]
                    ),
                    "decoded_pitch": float(
                        brain_out[
                            "decoded"
                        ][1]
                    ),
                    "decoded_yaw": float(
                        brain_out[
                            "decoded"
                        ][2]
                    ),
                    "ctrl_x": float(
                        data.ctrl[
                            moment_ids[0]
                        ]
                    ),
                    "ctrl_y": float(
                        data.ctrl[
                            moment_ids[1]
                        ]
                    ),
                    "ctrl_z": float(
                        data.ctrl[
                            moment_ids[2]
                        ]
                    ),
                    "thrust": float(
                        data.ctrl[
                            thrust_id
                        ]
                    ),
                    "ncon": int(
                        data.ncon
                    ),
                })

            if (
                viewer is not None
                and step
                % VIEWER_SYNC_EVERY
                == 0
            ):
                viewer.sync()

            if float(
                data.qpos[2]
            ) < -0.20:
                print(
                    "Emergency stop: "
                    "z < -0.20 m"
                )
                break

    if SHOW_VIEWER:
        try:
            from mujoco import (
                viewer as mj_viewer
            )

            viewer = (
                mj_viewer.launch_passive(
                    model,
                    data,
                )
            )

            try:
                simulation_loop(
                    viewer
                )
            finally:
                viewer.close()

        except Exception as exc:
            print(
                "Viewer unavailable, "
                "continuing headless:",
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

    final_gyro = (
        get_named_sensor(
            model,
            data,
            "body_gyro",
        )
    )

    final_euler = (
        quat_to_euler_wxyz(
            data.qpos[
                3:7
            ]
        )
    )

    final_roll_deg = (
        math.degrees(
            final_euler[0]
        )
    )

    final_pitch_deg = (
        math.degrees(
            final_euler[1]
        )
    )

    final_yaw_deg = (
        math.degrees(
            final_euler[2]
        )
    )

    final_gyro_norm = float(
        np.linalg.norm(
            final_gyro
        )
    )

    success = bool(
        not touched_ground
        and float(
            data.qpos[2]
        ) > 0.50
        and abs(
            final_roll_deg
        ) < 6.0
        and abs(
            final_pitch_deg
        ) < 6.0
        and abs(
            final_yaw_deg
        ) < 12.0
        and final_gyro_norm
        < 0.25
    )

    decoder_payload = {
        "training_neutral": (
            controller.training_neutral
            .tolist()
        ),
        "runtime_neutral": (
            controller.runtime_neutral
            .tolist()
        ),
        "feature_mean": (
            controller.feature_mean
            .tolist()
        ),
        "feature_std": (
            controller.feature_std
            .tolist()
        ),
        "regression": (
            controller.regression
            .tolist()
        ),
        "training_rmse": (
            controller.training_rmse
            .tolist()
        ),
        "hover_ctrl": (
            hover_ctrl
        ),
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

    summary = (
        "NEUROFLIGHT 3D ATTITUDE DEMO V2\n"
        f"PASS:               {success}\n"
        f"Final roll:         {final_roll_deg:+.3f} deg\n"
        f"Final pitch:        {final_pitch_deg:+.3f} deg\n"
        f"Final yaw:          {final_yaw_deg:+.3f} deg\n"
        f"Final gyro norm:    {final_gyro_norm:.4f} rad/s\n"
        f"Final position:     x={data.qpos[0]:+.3f}, "
        f"y={data.qpos[1]:+.3f}, z={data.qpos[2]:+.3f} m\n"
        f"Minimum z:          {min_z:.3f} m\n"
        f"Maximum tilt:       {max_tilt_deg:.2f} deg\n"
        f"Maximum gyro norm:  {max_gyro:.3f} rad/s\n"
        f"Ground touched:     {touched_ground}\n"
        f"Decoder train RMSE: {controller.training_rmse.tolist()}\n"
        "\n"
        "Control attribution:\n"
        "- roll/pitch/yaw moment command: MaleCNS computational model;\n"
        "- absolute level/heading reference: engineering attitude adapter;\n"
        "- altitude/thrust: classical engineering adapter.\n"
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
    print(
        summary
    )

    print(
        "Files:"
    )

    print(
        SUMMARY_TXT
    )

    print(
        FLIGHT_CSV
    )

    print(
        DECODER_JSON
    )


if __name__ == "__main__":
    run_demo()
