"""First protected 3-D flight test using the measured calibration matrix."""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from scripts.demo_neuroflight_3d_v2 import (
    NeuroFlight3AxisControllerV2, build_vehicle, clipped_ctrl,
    euler_to_quat_wxyz, get_named_sensor, quat_to_euler_wxyz, altitude_thrust,
)

# This is a deliberately conservative protected test, not a final controller.
START_Z = 1.50
SIM_SECONDS = 6.0
INITIAL_EULER_DEG = np.array([5.0, -4.0, 4.0])
INITIAL_BODY_RATES = np.array([0.10, -0.08, 0.08])
ATTITUDE_K = np.array([1.10, 0.85, 0.60])
MOMENT_GAIN = np.array([0.34, 0.20, 0.22])
OUTPUT_TAU_SECONDS = 0.22
MAX_ESTIMATED_RATE = np.array([0.55, 0.40, 0.40])
MAX_TILT_DEG, MIN_Z = 45.0, 0.70

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = Path("logs/demo_3d_v4") / stamp
OUT.mkdir(parents=True, exist_ok=True)


def newest_calibration() -> Path:
    files = sorted(Path("logs/calibration_3d").glob("*/calibration_3d.json"))
    if not files:
        raise RuntimeError("Calibration missing: run python -m scripts.calibrate_neuroflight_3d first")
    return files[-1]


def main() -> None:
    calibration_file = newest_calibration()
    cal = json.loads(calibration_file.read_text(encoding="utf-8"))
    if not cal.get("pass", False):
        raise RuntimeError("Latest calibration is not accepted; flight test cancelled")
    neutral = np.asarray(cal["neutral_output"], dtype=float)
    inverse = np.asarray(cal["decoder_matrix_input_from_centered_output"], dtype=float)

    controller = NeuroFlight3AxisControllerV2(seed=4242, verbose=True)
    model, data, thrust_id, moment_ids, ctrl_sign, hover_ctrl = build_vehicle()
    dt = float(model.opt.timestep)
    lif_steps = max(1, int(round(dt * 1000.0 / controller.brain.dt_ms)))
    alpha = math.exp(-dt / OUTPUT_TAU_SECONDS)

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, START_Z]
    q = np.radians(INITIAL_EULER_DEG)
    data.qpos[3:7] = euler_to_quat_wxyz(*q)
    data.qvel[:] = 0.0
    data.qvel[3:6] = INITIAL_BODY_RATES
    data.ctrl[thrust_id] = hover_ctrl
    mujoco.mj_forward(model, data)
    controller.reset_runtime(seed=999)

    estimate_filter = np.zeros(3)
    rows, failure = [], None
    max_tilt, max_gyro, min_z = 0.0, 0.0, START_Z
    steps = int(round(SIM_SECONDS / dt))
    log_stride = max(1, int(round(0.05 / dt)))

    for step in range(steps):
        gyro = get_named_sensor(model, data, "body_gyro")
        euler = quat_to_euler_wxyz(data.qpos[3:7])
        effective = gyro + ATTITUDE_K * euler
        brain = controller.step_from_effective_rate(effective, lif_steps)

        # The calibration was measured directly on this decoded vector.
        estimate = inverse @ (brain["decoded"] - neutral)
        estimate = np.clip(estimate, -MAX_ESTIMATED_RATE, MAX_ESTIMATED_RATE)
        estimate_filter = alpha * estimate_filter + (1.0 - alpha) * estimate
        moment = ctrl_sign * MOMENT_GAIN * estimate_filter

        z, vz = float(data.qpos[2]), float(data.qvel[2])
        thrust = altitude_thrust(model, thrust_id, hover_ctrl, z, vz, START_Z, 0.0, euler[0], euler[1])
        data.ctrl[:] = 0.0
        data.ctrl[thrust_id] = thrust
        for axis, actuator_id in enumerate(moment_ids):
            data.ctrl[actuator_id] = clipped_ctrl(model, actuator_id, moment[axis])
        mujoco.mj_step(model, data)

        gyro_after = get_named_sensor(model, data, "body_gyro")
        euler_after = quat_to_euler_wxyz(data.qpos[3:7])
        tilt = max(abs(math.degrees(euler_after[0])), abs(math.degrees(euler_after[1])))
        gyro_norm = float(np.linalg.norm(gyro_after))
        min_z, max_tilt, max_gyro = min(min_z, float(data.qpos[2])), max(max_tilt, tilt), max(max_gyro, gyro_norm)
        if step % log_stride == 0:
            rows.append({"time": data.time, "z": float(data.qpos[2]), "roll_deg": math.degrees(euler_after[0]), "pitch_deg": math.degrees(euler_after[1]), "yaw_deg": math.degrees(euler_after[2]), "gyro_norm": gyro_norm, "decoded_roll": brain["decoded"][0], "decoded_pitch": brain["decoded"][1], "decoded_yaw": brain["decoded"][2], "estimate_roll": estimate_filter[0], "estimate_pitch": estimate_filter[1], "estimate_yaw": estimate_filter[2], "ctrl_roll": moment[0], "ctrl_pitch": moment[1], "ctrl_yaw": moment[2], "thrust": thrust})
        if float(data.qpos[2]) < MIN_Z:
            failure = f"altitude below {MIN_Z:.2f} m"; break
        if tilt > MAX_TILT_DEG:
            failure = f"tilt above {MAX_TILT_DEG:.0f} deg"; break

    with (OUT / "flight.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    final_euler = quat_to_euler_wxyz(data.qpos[3:7])
    passed = failure is None and max_tilt < 20.0 and min_z > 1.10 and float(np.linalg.norm(get_named_sensor(model, data, "body_gyro"))) < 0.35
    summary = f"""NEUROFLIGHT 3D DEMO V4\nPASS:               {passed}\nFailure reason:     {failure or 'none'}\nCalibration:        {calibration_file}\nFinal roll:         {math.degrees(final_euler[0]):+.3f} deg\nFinal pitch:        {math.degrees(final_euler[1]):+.3f} deg\nFinal yaw:          {math.degrees(final_euler[2]):+.3f} deg\nFinal gyro norm:    {np.linalg.norm(get_named_sensor(model, data, 'body_gyro')):.4f} rad/s\nMinimum z:          {min_z:.3f} m\nMaximum tilt:       {max_tilt:.2f} deg\nMaximum gyro norm:  {max_gyro:.3f} rad/s\n"""
    (OUT / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Files: {OUT / 'SUMMARY.txt'} ; {OUT / 'flight.csv'}")


if __name__ == "__main__":
    main()
