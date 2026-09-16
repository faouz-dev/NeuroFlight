from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import math
import sys

import mujoco
import mujoco_menagerie as mm
import numpy as np

from scripts.demo_neuroflight_3d import (
    NeuroFlight3AxisController,
    get_actuator_id,
    clipped_ctrl,
    get_named_sensor,
    quat_to_euler_wxyz,
    MOTOR_GAIN,
)


# ============================================================
# NEUROFLIGHT - LANDING DEMO
#
# MaleCNS:
#   controls roll + pitch + yaw body moments.
#
# Vertical landing channel:
#   is deliberately a simple classical altitude/thrust adapter.
#   It is NOT claimed to be MaleCNS altitude control.
#
# This lets us demonstrate a real touchdown now, without pretending
# that an altitude/visual circuit has already been mapped biologically.
# ============================================================


START_Z = 1.20
HOLD_SECONDS = 1.0
DESCENT_RATE = 0.14          # m/s
LANDING_TARGET_Z = 0.055     # approximate center height near floor
MAX_SECONDS = 12.0

# Small 3-axis perturbation at the start.
INITIAL_BODY_RATES = np.array(
    [0.22, -0.18, 0.14],
    dtype=float,
)

# Vertical PD in force units (N per unit error).
KP_Z = 0.12
KD_Z = 0.075

# Touchdown criteria.
TOUCHDOWN_Z_LIMIT = 0.15
TOUCHDOWN_MAX_VZ = 0.60

SHOW_VIEWER = "--headless" not in sys.argv
VIEWER_SYNC_EVERY = 5

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path("logs/landing") / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_CSV = OUT_DIR / "landing.csv"
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


def any_low_contact(data):
    return bool(
        data.ncon > 0
        and float(data.qpos[2]) < TOUCHDOWN_Z_LIMIT
    )


def run_landing():
    print()
    print("=" * 78)
    print("NEUROFLIGHT LANDING DEMO")
    print("=" * 78)

    controller = NeuroFlight3AxisController(
        seed=5252,
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
        seed=888,
    )

    print(
        "Hover ctrl:",
        f"{hover_ctrl:.6f}",
    )

    print(
        "Start z:",
        START_Z,
    )

    print(
        "Descent rate:",
        DESCENT_RATE,
        "m/s",
    )

    print(
        "Viewer:",
        "ON" if SHOW_VIEWER else "HEADLESS",
    )

    total_steps = int(
        round(
            MAX_SECONDS
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

    touchdown = False
    touchdown_time = None
    touchdown_vz = None
    touchdown_gyro_norm = None

    thrust_after_touchdown = hover_ctrl

    def simulation_loop(viewer=None):
        nonlocal touchdown
        nonlocal touchdown_time
        nonlocal touchdown_vz
        nonlocal touchdown_gyro_norm
        nonlocal thrust_after_touchdown

        for step in range(total_steps):
            t = float(data.time)

            gyro = get_named_sensor(
                model,
                data,
                "body_gyro",
            )

            brain_out = controller.step_from_gyro(
                gyro,
                lif_steps,
            )

            moment_ctrl = (
                ctrl_sign
                * MOTOR_GAIN
                * brain_out["decoded"]
            )

            z = float(
                data.qpos[2]
            )

            vz = float(
                data.qvel[2]
            )

            # ----------------------------
            # Landing trajectory
            # ----------------------------

            if t < HOLD_SECONDS:
                target_z = START_Z
                target_vz = 0.0
            else:
                target_z = max(
                    LANDING_TARGET_Z,
                    START_Z
                    - DESCENT_RATE
                    * (t - HOLD_SECONDS),
                )

                target_vz = (
                    -DESCENT_RATE
                    if target_z > LANDING_TARGET_Z + 1e-6
                    else 0.0
                )

            z_error = (
                target_z
                - z
            )

            vz_error = (
                target_vz
                - vz
            )

            thrust_cmd = (
                hover_ctrl
                + KP_Z * z_error
                + KD_Z * vz_error
            )

            # Once physical contact near the floor is observed,
            # progressively remove thrust.
            if (
                not touchdown
                and any_low_contact(data)
            ):
                touchdown = True
                touchdown_time = t
                touchdown_vz = vz
                touchdown_gyro_norm = float(
                    np.linalg.norm(
                        gyro
                    )
                )

                thrust_after_touchdown = float(
                    data.ctrl[thrust_id]
                )

                print()
                print(
                    "TOUCHDOWN at",
                    f"{t:.3f}s",
                    "| z=",
                    f"{z:.3f}",
                    "| vz=",
                    f"{vz:+.3f}",
                )

            if touchdown:
                # Ramp down over roughly 0.6 s after contact.
                elapsed = max(
                    0.0,
                    t - touchdown_time,
                )

                factor = max(
                    0.0,
                    1.0
                    - elapsed / 0.60,
                )

                thrust_cmd = (
                    thrust_after_touchdown
                    * factor
                )

            thrust_cmd = clipped_ctrl(
                model,
                thrust_id,
                thrust_cmd,
            )

            data.ctrl[:] = 0.0
            data.ctrl[thrust_id] = thrust_cmd

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

            if step % log_every == 0:
                rows.append({
                    "time": float(data.time),
                    "x": float(data.qpos[0]),
                    "y": float(data.qpos[1]),
                    "z": float(data.qpos[2]),
                    "target_z": float(target_z),
                    "vz": float(data.qvel[2]),
                    "target_vz": float(target_vz),
                    "roll_deg": math.degrees(euler[0]),
                    "pitch_deg": math.degrees(euler[1]),
                    "yaw_deg": math.degrees(euler[2]),
                    "gyro_x": float(gyro_after[0]),
                    "gyro_y": float(gyro_after[1]),
                    "gyro_z": float(gyro_after[2]),
                    "gyro_norm": float(
                        np.linalg.norm(
                            gyro_after
                        )
                    ),
                    "decoded_roll": float(brain_out["decoded"][0]),
                    "decoded_pitch": float(brain_out["decoded"][1]),
                    "decoded_yaw": float(brain_out["decoded"][2]),
                    "ctrl_x": float(data.ctrl[moment_ids[0]]),
                    "ctrl_y": float(data.ctrl[moment_ids[1]]),
                    "ctrl_z": float(data.ctrl[moment_ids[2]]),
                    "thrust": float(data.ctrl[thrust_id]),
                    "ncon": int(data.ncon),
                    "touchdown": int(touchdown),
                })

            if viewer is not None and step % VIEWER_SYNC_EVERY == 0:
                viewer.sync()

            # End after thrust has been removed and vehicle has had time to settle.
            if (
                touchdown
                and t - touchdown_time >= 1.25
            ):
                break

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

    final_gyro_norm = float(
        np.linalg.norm(
            final_gyro
        )
    )

    landing_soft = bool(
        touchdown
        and touchdown_vz is not None
        and abs(touchdown_vz) <= TOUCHDOWN_MAX_VZ
    )

    summary = (
        "NEUROFLIGHT LANDING DEMO\n"
        f"Touchdown detected: {touchdown}\n"
        f"Touchdown time:     {touchdown_time}\n"
        f"Touchdown vz:       {touchdown_vz}\n"
        f"Touchdown gyro:     {touchdown_gyro_norm}\n"
        f"Soft touchdown:     {landing_soft}\n"
        f"Final z:            {float(data.qpos[2]):.4f} m\n"
        f"Final gyro norm:    {final_gyro_norm:.4f} rad/s\n"
        f"Final contacts:     {int(data.ncon)}\n"
        "\n"
        "Control attribution:\n"
        "- Roll/pitch/yaw body moments: MaleCNS computational controller.\n"
        "- Vertical descent/thrust: classical PD adapter, not MaleCNS.\n"
        "\n"
        "This demo therefore validates integration and touchdown, not yet a "
        "biological MaleCNS altitude/landing circuit.\n"
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


if __name__ == "__main__":
    run_landing()
