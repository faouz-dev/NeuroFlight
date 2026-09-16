from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import json
import math
import sys

import mujoco
import numpy as np

from scripts.demo_neuroflight_3d_v2 import (
    NeuroFlight3AxisControllerV2,
    build_vehicle,
    get_named_sensor,
    quat_to_euler_wxyz,
    euler_to_quat_wxyz,
    clipped_ctrl,
    altitude_thrust,
)


# ============================================================
# NEUROFLIGHT - 3D DEMO V3
#
# This version fixes what the previous real 3D run exposed:
#
# 1) automatic sign calibration for each neural output
#    (the previous run showed the combined yaw output inverted);
#
# 2) automatic actuator gain calibration for each axis;
#
# 3) stronger but explicit engineering attitude reference so the
#    aircraft does not spend several seconds at 30+ degrees of tilt;
#
# 4) fail-fast before a ground impact can be misread as stabilization.
#
# MaleCNS remains the source of the roll/pitch/yaw motor signal.
# Absolute attitude and altitude remain engineering adapters.
# ============================================================


ATTITUDE_K = np.array(
    [2.8, 2.8, 1.6],
    dtype=float,
)

INITIAL_EULER_DEG = np.array(
    [8.0, -6.0, 7.0],
    dtype=float,
)

INITIAL_BODY_RATES = np.array(
    [0.20, -0.16, 0.12],
    dtype=float,
)

START_Z = 1.50
SIM_SECONDS = 7.0

# Adapter calibration.
PROBE_EFFECTIVE_RATE = 1.0
PROBE_SECONDS = 0.70
PROBE_IGNORE_SECONDS = 0.30
TARGET_CTRL_AT_PROBE = 0.80
MIN_AXIS_GAIN = 0.75
MAX_AXIS_GAIN = 4.50

# Safety / verdict.
FAIL_Z = 0.40
FAIL_TILT_DEG = 70.0

PASS_ROLL_DEG = 6.0
PASS_PITCH_DEG = 6.0
PASS_YAW_DEG = 10.0
PASS_GYRO_NORM = 0.25
PASS_MIN_Z = 0.80

SHOW_VIEWER = "--headless" not in sys.argv
VIEWER_SYNC_EVERY = 5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/demo_3d_v3") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_CSV = OUT_DIR / "flight.csv"
ADAPTER_JSON = OUT_DIR / "adapter_calibration.json"
SUMMARY_TXT = OUT_DIR / "SUMMARY.txt"


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


def calibrate_axis_adapter(
    controller,
    physics_dt,
    lif_steps,
):
    """
    Measures the actual sign and amplitude of the already-trained
    combined neural readout for +/- excitation of each axis.

    It calibrates only the fly-motor -> drone actuator adapter.
    """

    total_steps = int(
        round(
            PROBE_SECONDS
            / physics_dt
        )
    )

    ignore_steps = int(
        round(
            PROBE_IGNORE_SECONDS
            / physics_dt
        )
    )

    plus_mean = np.zeros(
        3,
        dtype=float,
    )

    minus_mean = np.zeros(
        3,
        dtype=float,
    )

    def probe(axis, direction, seed):
        controller.reset_runtime(
            seed=seed
        )

        effective = np.zeros(
            3,
            dtype=float,
        )

        effective[axis] = (
            direction
            * PROBE_EFFECTIVE_RATE
        )

        values = []

        for step in range(
            total_steps
        ):
            out = (
                controller.step_from_effective_rate(
                    effective,
                    lif_steps,
                )
            )

            if step >= ignore_steps:
                values.append(
                    float(
                        out[
                            "decoded"
                        ][axis]
                    )
                )

        return float(
            np.mean(
                values
            )
        )

    for axis in range(3):
        plus_mean[axis] = probe(
            axis,
            +1.0,
            7000 + axis * 100,
        )

        minus_mean[axis] = probe(
            axis,
            -1.0,
            7050 + axis * 100,
        )

    amplitude = (
        plus_mean
        - minus_mean
    ) / 2.0

    offset = (
        plus_mean
        + minus_mean
    ) / 2.0

    output_sign = np.where(
        amplitude >= 0.0,
        1.0,
        -1.0,
    )

    oriented_amplitude = np.abs(
        amplitude
    )

    gain = np.clip(
        TARGET_CTRL_AT_PROBE
        / np.maximum(
            oriented_amplitude,
            1e-3,
        ),
        MIN_AXIS_GAIN,
        MAX_AXIS_GAIN,
    )

    return {
        "plus_mean": plus_mean,
        "minus_mean": minus_mean,
        "amplitude": amplitude,
        "offset": offset,
        "output_sign": output_sign,
        "gain": gain,
    }


def run_demo():
    print()
    print("=" * 78)
    print("NEUROFLIGHT 3D DEMO V3")
    print("=" * 78)

    controller = NeuroFlight3AxisControllerV2(
        seed=4242,
        verbose=True,
    )

    (
        model,
        data,
        thrust_id,
        moment_ids,
        actuator_ctrl_sign,
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

    print()
    print(
        "Calibrating neural-output sign and actuator gain..."
    )

    adapter = calibrate_axis_adapter(
        controller,
        physics_dt,
        lif_steps,
    )

    axis_names = [
        "roll",
        "pitch",
        "yaw",
    ]

    print()

    for axis, name in enumerate(
        axis_names
    ):
        print(
            f"{name:5s} "
            f"| neural(-)={adapter['minus_mean'][axis]:+.4f} "
            f"| neural(+)={adapter['plus_mean'][axis]:+.4f} "
            f"| sign={adapter['output_sign'][axis]:+.0f} "
            f"| gain={adapter['gain'][axis]:.3f}"
        )

    with open(
        ADAPTER_JSON,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                key: (
                    value.tolist()
                    if isinstance(
                        value,
                        np.ndarray,
                    )
                    else value
                )
                for key, value
                in adapter.items()
            },
            file,
            indent=2,
        )

    # Fresh runtime state after calibration.
    controller.reset_runtime(
        seed=999,
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

    initial_euler = np.radians(
        INITIAL_EULER_DEG
    )

    data.qpos[3:7] = (
        euler_to_quat_wxyz(
            initial_euler[0],
            initial_euler[1],
            initial_euler[2],
        )
    )

    data.qvel[:] = 0.0

    data.qvel[3:6] = (
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

    min_z = START_Z
    max_tilt = 0.0
    max_gyro = 0.0
    failure_reason = None

    def loop(viewer=None):
        nonlocal min_z
        nonlocal max_tilt
        nonlocal max_gyro
        nonlocal failure_reason

        for step in range(
            total_steps
        ):
            gyro_xyz = get_named_sensor(
                model,
                data,
                "body_gyro",
            )

            euler_xyz = (
                quat_to_euler_wxyz(
                    data.qpos[3:7]
                )
            )

            # Engineering attitude reference feeding the neural
            # sensor adapter.
            effective_rate = (
                gyro_xyz
                + ATTITUDE_K
                * euler_xyz
            )

            brain_out = (
                controller.step_from_effective_rate(
                    effective_rate,
                    lif_steps,
                )
            )

            # Correct session-measured output offset/sign and scale it
            # into Crazyflie actuator coordinates.
            corrected_neural = (
                adapter[
                    "output_sign"
                ]
                * (
                    brain_out[
                        "decoded"
                    ]
                    - adapter[
                        "offset"
                    ]
                )
            )

            moment_ctrl = (
                actuator_ctrl_sign
                * adapter[
                    "gain"
                ]
                * corrected_neural
            )

            moment_ctrl = np.clip(
                moment_ctrl,
                -1.0,
                1.0,
            )

            z = float(
                data.qpos[2]
            )

            vz = float(
                data.qvel[2]
            )

            thrust_cmd = altitude_thrust(
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

            gyro_after = get_named_sensor(
                model,
                data,
                "body_gyro",
            )

            euler_after = (
                quat_to_euler_wxyz(
                    data.qpos[3:7]
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

            min_z = min(
                min_z,
                float(
                    data.qpos[2]
                ),
            )

            max_tilt = max(
                max_tilt,
                tilt_deg,
            )

            max_gyro = max(
                max_gyro,
                gyro_norm,
            )

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
                    "roll_deg": math.degrees(
                        euler_after[0]
                    ),
                    "pitch_deg": math.degrees(
                        euler_after[1]
                    ),
                    "yaw_deg": math.degrees(
                        euler_after[2]
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
                    "gyro_norm": gyro_norm,
                    "effective_x": float(
                        effective_rate[0]
                    ),
                    "effective_y": float(
                        effective_rate[1]
                    ),
                    "effective_z": float(
                        effective_rate[2]
                    ),
                    "raw_decoded_roll": float(
                        brain_out[
                            "decoded"
                        ][0]
                    ),
                    "raw_decoded_pitch": float(
                        brain_out[
                            "decoded"
                        ][1]
                    ),
                    "raw_decoded_yaw": float(
                        brain_out[
                            "decoded"
                        ][2]
                    ),
                    "corrected_roll": float(
                        corrected_neural[0]
                    ),
                    "corrected_pitch": float(
                        corrected_neural[1]
                    ),
                    "corrected_yaw": float(
                        corrected_neural[2]
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
            ) < FAIL_Z:
                failure_reason = (
                    f"altitude below "
                    f"{FAIL_Z:.2f} m"
                )

                print(
                    "FAIL-FAST:",
                    failure_reason,
                )

                break

            if tilt_deg > FAIL_TILT_DEG:
                failure_reason = (
                    f"tilt above "
                    f"{FAIL_TILT_DEG:.1f} deg"
                )

                print(
                    "FAIL-FAST:",
                    failure_reason,
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
                loop(
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

            loop(
                None
            )
    else:
        loop(
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

    final_euler = (
        quat_to_euler_wxyz(
            data.qpos[3:7]
        )
    )

    final_roll = math.degrees(
        final_euler[0]
    )

    final_pitch = math.degrees(
        final_euler[1]
    )

    final_yaw = math.degrees(
        final_euler[2]
    )

    final_gyro_norm = float(
        np.linalg.norm(
            final_gyro
        )
    )

    passed = bool(
        failure_reason is None
        and float(
            data.qpos[2]
        ) >= PASS_MIN_Z
        and abs(
            final_roll
        ) <= PASS_ROLL_DEG
        and abs(
            final_pitch
        ) <= PASS_PITCH_DEG
        and abs(
            final_yaw
        ) <= PASS_YAW_DEG
        and final_gyro_norm
        <= PASS_GYRO_NORM
    )

    summary = (
        "NEUROFLIGHT 3D DEMO V3\n"
        f"PASS:               {passed}\n"
        f"Failure reason:     {failure_reason}\n"
        f"Final roll:         {final_roll:+.3f} deg\n"
        f"Final pitch:        {final_pitch:+.3f} deg\n"
        f"Final yaw:          {final_yaw:+.3f} deg\n"
        f"Final gyro norm:    {final_gyro_norm:.4f} rad/s\n"
        f"Final position:     "
        f"x={data.qpos[0]:+.3f}, "
        f"y={data.qpos[1]:+.3f}, "
        f"z={data.qpos[2]:+.3f} m\n"
        f"Minimum z:          {min_z:.3f} m\n"
        f"Maximum tilt:       {max_tilt:.2f} deg\n"
        f"Maximum gyro norm:  {max_gyro:.3f} rad/s\n"
        f"Adapter signs:      "
        f"{adapter['output_sign'].tolist()}\n"
        f"Adapter gains:      "
        f"{adapter['gain'].tolist()}\n"
        "\n"
        "Attribution:\n"
        "- body-moment signal: MaleCNS computational model;\n"
        "- neural-output sign/gain: calibrated fly-to-drone adapter;\n"
        "- absolute attitude reference: engineering adapter;\n"
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
        ADAPTER_JSON
    )


if __name__ == "__main__":
    run_demo()
