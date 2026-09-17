"""Regression checks for signed physical torque and finite readout scaling."""
import unittest
import numpy as np
import mujoco
from scripts.roll_closed_loop import select_model
from scripts.demo_neuroflight_3d_v2 import build_vehicle


class AdapterTests(unittest.TestCase):
    def test_scaling_floor(self):
        model=select_model(robust=True)['model']
        self.assertTrue(np.all(np.asarray(model['scale'])>=.5))
        self.assertTrue(np.isfinite(model['weights']).all())

    def test_physical_torque_opposes_signed_error(self):
        m,d,_,mids,_,_=build_vehicle()
        for error in (-.2,.2):
            mujoco.mj_resetData(m,d);d.qpos[2]=1.5
            torque=-1e-5*error
            d.ctrl[mids[0]]=torque/m.actuator_gear[mids[0],3]
            mujoco.mj_forward(m,d)
            self.assertLess(d.qacc[3]*error,0)

    def test_neural_clock_covers_physics_step(self):
        m,*_=build_vehicle()
        steps=round(m.opt.timestep/.0005)
        self.assertGreaterEqual(steps,1)
        self.assertAlmostEqual(steps*.0005,m.opt.timestep)


if __name__=='__main__':unittest.main()
