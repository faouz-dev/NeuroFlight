"""Protected 3-D test: session-neutral MaleCNS calibration and tracking camera."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime
from pathlib import Path
import mujoco, mujoco.viewer
import numpy as np
from scripts.demo_neuroflight_3d_v2 import NeuroFlight3AxisControllerV2, build_vehicle, clipped_ctrl, euler_to_quat_wxyz, get_named_sensor, quat_to_euler_wxyz, altitude_thrust

START_Z=1.50; SIM_SECONDS=6.0
INITIAL_EULER_DEG=np.array([5.0,-4.0,4.0]); INITIAL_BODY_RATES=np.array([.10,-.08,.08])
ATTITUDE_K=np.array([1.10,.85,.60]); MOMENT_GAIN=np.array([.34,.20,.22])
OUTPUT_TAU=0.22; MAX_EST=np.array([.55,.40,.40]); MIN_Z=.70; MAX_TILT=45.0
WARMUP_MS=600.; NEUTRAL_MS=300.
OUT=Path('logs/demo_3d_v5')/datetime.now().strftime('%Y%m%d_%H%M%S'); OUT.mkdir(parents=True,exist_ok=True)

def latest_calibration():
    files=sorted(Path('logs/calibration_3d').glob('*/calibration_3d.json'))
    if not files: raise RuntimeError('Run scripts.calibrate_neuroflight_3d first.')
    data=json.loads(files[-1].read_text(encoding='utf-8'))
    if not data.get('pass'): raise RuntimeError('Latest calibration was not accepted.')
    return files[-1],np.asarray(data['decoder_matrix_input_from_centered_output'],float)

def session_neutral(controller):
    """Measure this stochastic session's decoded zero after its filters settle."""
    values=[]; steps=int(round((WARMUP_MS+NEUTRAL_MS)/controller.brain.dt_ms)); begin=int(round(WARMUP_MS/controller.brain.dt_ms))
    for i in range(steps):
        out=controller.step_from_effective_rate(np.zeros(3),1)['decoded']
        if i>=begin: values.append(out.copy())
    return np.mean(values,axis=0)

def run(viewer=None, vehicle=None):
    cal_path,inverse=latest_calibration(); controller=NeuroFlight3AxisControllerV2(seed=4242,verbose=True)
    model,data,thrust_id,moment_ids,ctrl_sign,hover = vehicle or build_vehicle(); dt=float(model.opt.timestep); alpha=math.exp(-dt/OUTPUT_TAU)
    steps_ratio = dt / (controller.brain.dt_ms / 1000.0)
    lif_steps = int(round(steps_ratio))
    if lif_steps < 1 or not math.isclose(steps_ratio, lif_steps, abs_tol=1e-9):
        raise ValueError('Physics timestep must be an integer multiple of the LIF timestep.')
    controller.reset_runtime(seed=999); neutral=session_neutral(controller)
    print('Session neutral:',np.array2string(neutral,precision=4)); print('Calibration:',cal_path)
    mujoco.mj_resetData(model,data); data.qpos[:3]=[0,0,START_Z]; data.qpos[3:7]=euler_to_quat_wxyz(*np.radians(INITIAL_EULER_DEG)); data.qvel[:]=0; data.qvel[3:6]=INITIAL_BODY_RATES; mujoco.mj_forward(model,data)
    filt=np.zeros(3); rows=[]; failure=None; min_z=START_Z; max_tilt=max_gyro=0.
    for step in range(int(round(SIM_SECONDS/dt))):
        gyro=get_named_sensor(model,data,'body_gyro'); euler=quat_to_euler_wxyz(data.qpos[3:7]); effective=gyro+ATTITUDE_K*euler
        # Advance both neural dynamics and its readout filter in physical time.
        for _ in range(lif_steps):
            decoded=controller.step_from_effective_rate(effective,1)['decoded']
        estimate=np.clip(inverse@(decoded-neutral),-MAX_EST,MAX_EST); filt=alpha*filt+(1-alpha)*estimate
        # build_vehicle returns -sign(gear); this already supplies the restoring sign.
        moment = ctrl_sign * MOMENT_GAIN * filt
        thrust=altitude_thrust(model,thrust_id,hover,float(data.qpos[2]),float(data.qvel[2]),START_Z,0.,euler[0],euler[1]); data.ctrl[:]=0; data.ctrl[thrust_id]=thrust
        for a,act in enumerate(moment_ids): data.ctrl[act]=clipped_ctrl(model,act,moment[a])
        mujoco.mj_step(model,data)
        after=get_named_sensor(model,data,'body_gyro'); ea=quat_to_euler_wxyz(data.qpos[3:7]); tilt=max(abs(math.degrees(ea[0])),abs(math.degrees(ea[1]))); gn=float(np.linalg.norm(after)); min_z=min(min_z,float(data.qpos[2])); max_tilt=max(max_tilt,tilt); max_gyro=max(max_gyro,gn)
        if viewer:
            viewer.cam.lookat[:]=data.qpos[:3]; viewer.cam.distance=2.4; viewer.cam.azimuth=135; viewer.cam.elevation=-18; viewer.sync()
        if step%max(1,int(round(.05/dt)))==0: rows.append({'time':data.time,'z':float(data.qpos[2]),'roll_deg':math.degrees(ea[0]),'pitch_deg':math.degrees(ea[1]),'yaw_deg':math.degrees(ea[2]),'gyro_norm':gn,'decoded_roll':decoded[0],'decoded_pitch':decoded[1],'decoded_yaw':decoded[2],'estimate_roll':filt[0],'estimate_pitch':filt[1],'estimate_yaw':filt[2],'ctrl_roll':moment[0],'ctrl_pitch':moment[1],'ctrl_yaw':moment[2],'thrust':thrust})
        if float(data.qpos[2])<MIN_Z: failure=f'altitude below {MIN_Z:.2f} m'; break
        if tilt>MAX_TILT: failure=f'tilt above {MAX_TILT:.0f} deg'; break
    with (OUT/'flight.csv').open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    final=quat_to_euler_wxyz(data.qpos[3:7]); passed=failure is None and max_tilt<20 and min_z>1.10 and float(np.linalg.norm(get_named_sensor(model,data,'body_gyro')))<.35
    text=f'''NEUROFLIGHT 3D DEMO V5\nPASS:               {passed}\nFailure reason:     {failure or 'none'}\nSession neutral:    {neutral.tolist()}\nCalibration:        {cal_path}\nFinal roll:         {math.degrees(final[0]):+.3f} deg\nFinal pitch:        {math.degrees(final[1]):+.3f} deg\nFinal yaw:          {math.degrees(final[2]):+.3f} deg\nFinal gyro norm:    {np.linalg.norm(get_named_sensor(model,data,'body_gyro')):.4f} rad/s\nMinimum z:          {min_z:.3f} m\nMaximum tilt:       {max_tilt:.2f} deg\nMaximum gyro norm:  {max_gyro:.3f} rad/s\n'''; (OUT/'SUMMARY.txt').write_text(text,encoding='utf-8'); print(text)

if __name__=='__main__':
    if '--headless' in sys.argv: run()
    else:
        # The look-at point is updated every simulation step, keeping the drone centered.
        vehicle=build_vehicle()
        with mujoco.viewer.launch_passive(vehicle[0],vehicle[1]) as viewer: run(viewer,vehicle)
