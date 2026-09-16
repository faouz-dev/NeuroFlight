"""MuJoCo-only regression check: positive estimated error needs negative torque."""
import mujoco
import numpy as np
from scripts.demo_neuroflight_3d_v2 import build_vehicle


def main():
    model, data, _, moment_ids, signs, _ = build_vehicle()
    ratio = model.opt.timestep / 0.0005
    assert np.isclose(ratio, round(ratio))
    print('LIF steps per physics step:', int(round(ratio)))
    for axis, actuator in enumerate(moment_ids):
        for error in (-0.1, 0.1):
            mujoco.mj_resetData(model, data)
            data.qpos[2] = 1.5
            data.ctrl[actuator] = signs[axis] * 0.2 * error
            mujoco.mj_forward(model, data)
            acceleration = data.qacc[3 + axis]
            assert acceleration * error < 0, (axis, error, acceleration)
            print('PASS', axis, error, acceleration)


if __name__ == '__main__':
    main()
