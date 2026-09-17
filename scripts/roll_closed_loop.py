"""Headless, isolated-roll experiment, not 3-D flight or biological validation.

Actual MaleCNS spikes -> motor readout -> estimated effective rate -> physical
roll torque. Attitude reference, encoding, torque scale and altitude hold are
engineering adapters. Controls: oracle PD, disconnected sensory input, no torque.
No gyro or attitude bypass reaches the neural torque command.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import mujoco
from scripts.compare_readout_v2 import RawMotors, arrays, features, fit, predict, LAMBDAS
from scripts.demo_neuroflight_3d_v2 import (
    build_vehicle, altitude_thrust, clipped_ctrl, euler_to_quat_wxyz,
    quat_to_euler_wxyz, get_named_sensor, GYRO_TO_HZ, DELTA_RATE_HZ)

ATTITUDE_K = 2.0
DAMPING = 3.0
OUTPUT_TAU = .05
DURATION = 4.0
SCENARIOS = ((12.,.15,1100001), (-12.,-.15,1100002),
             (20.,.25,1100003), (-20.,-.25,1100004))


def select_model(robust=False):
    base = Path('results/readout_v2')
    train = json.loads((base/'train.json').read_text())
    val = json.loads((base/'validation.json').read_text())
    choices = []
    for kind in ('legacy3','motor10'):
        x,y = arrays(train,kind); xv,yv = arrays(val,kind)
        for lam in LAMBDAS:
            if robust:
                # A floor in spike-trace units prevents near-silent channels
                # becoming huge gain paths when real-time activity returns.
                mean=x.mean(0);scale=np.maximum(x.std(0),.5)
                a=np.column_stack((np.ones(len(x)),(x-mean)/scale))
                reg=np.eye(a.shape[1])*lam;reg[0,0]=0
                w=np.linalg.solve(a.T@a+reg,a.T@y)
                m=dict(mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist())
            else:m=fit(x,y,lam)
            error=float(np.sqrt(np.mean((predict(m,xv)[:,0]-yv[:,0])**2)))
            choices.append(dict(kind=kind,penalty=lam,model=m,validation_roll_rmse=error,scale_floor=.5 if robust else 1e-6))
    return min(choices,key=lambda c:c['validation_roll_rmse'])


def decode(brain, candidate):
    raw=brain.traces[None,:]
    f=features(raw,np.zeros_like(raw),candidate['kind'])
    return float(predict(candidate['model'],f)[0,0])


def experiment(brain,candidate,vehicle,mode,angle,rate,seed):
    m,d,tid,mids,_,hover=vehicle
    dt=float(m.opt.timestep); steps=int(round(dt/(brain.brain.dt_ms/1000)))
    assert steps>=1 and math.isclose(steps*brain.brain.dt_ms/1000,dt)
    rng=np.random.default_rng(seed)
    brain._reset_brain_state()
    neutral=[]
    for i in range(round(600/brain.brain.dt_ms)):
        brain._tick(rng,np.zeros(3))
        if i>=round(400/brain.brain.dt_ms):neutral.append(decode(brain,candidate))
    offset=float(np.mean(neutral))
    mujoco.mj_resetData(m,d)
    d.qpos[:3]=[0,0,1.5]
    d.qpos[3:7]=euler_to_quat_wxyz(math.radians(angle),0.,0.)
    d.qvel[3]=rate
    mujoco.mj_forward(m,d)
    inertia=np.zeros((m.nv,m.nv))
    mujoco.mj_fullM(m,d,inertia)
    roll_inertia=float(inertia[3,3])
    gear=float(m.actuator_gear[mids[0],3]); assert abs(gear)>1e-12
    alpha=math.exp(-dt/OUTPUT_TAU)
    filt=0.; rows=[]; reason=None
    for tick in range(round(DURATION/dt)):
        euler=quat_to_euler_wxyz(d.qpos[3:7])
        gyro=get_named_sensor(m,d,'body_gyro')
        effective=float(gyro[0]+ATTITUDE_K*euler[0])
        if mode in ('neural','disconnected'):
            sensory=effective if mode=='neural' else 0.
            command=np.array([np.clip(sensory*GYRO_TO_HZ/DELTA_RATE_HZ,-1,1),0.,0.])
            for _ in range(steps):brain._tick(rng,command)
            estimate=float(np.clip(decode(brain,candidate)-offset,-1.5,1.5))
            filt=alpha*filt+(1-alpha)*estimate
        elif mode=='oracle':filt=effective
        elif mode=='none':filt=0.
        else:raise ValueError(mode)
        torque=-roll_inertia*DAMPING*filt
        d.ctrl[:]=0
        d.ctrl[tid]=altitude_thrust(m,tid,hover,float(d.qpos[2]),float(d.qvel[2]),1.5,0.,euler[0],euler[1])
        d.ctrl[mids[0]]=clipped_ctrl(m,mids[0],torque/gear)
        mujoco.mj_step(m,d)
        if tick%5==0:
            after=quat_to_euler_wxyz(d.qpos[3:7])
            rows.append(dict(t=float(d.time),roll_deg=math.degrees(after[0]),
                             rate=float(d.qvel[3]),z=float(d.qpos[2]),
                             estimated_effective_rate=filt,desired_torque_nm=torque))
        roll=math.degrees(quat_to_euler_wxyz(d.qpos[3:7])[0])
        if abs(roll)>60 or d.qpos[2]<.5 or not np.isfinite(d.qpos).all():
            reason='tilt, altitude or finite-state bound';break
    tail=[r for r in rows if r['t']>=DURATION-.5]
    mean_abs=float(np.mean([abs(r['roll_deg']) for r in tail])) if tail else None
    final_roll=math.degrees(quat_to_euler_wxyz(d.qpos[3:7])[0])
    final_rate=float(d.qvel[3])
    passed=reason is None and mean_abs is not None and mean_abs<5 and abs(final_rate)<.2
    return dict(mode=mode,initial_roll_deg=angle,initial_rate=rate,seed=seed,
                pass_criteria='last 0.5s mean |roll| <5deg, final |rate| <0.2rad/s, full duration within bounds',
                passed=bool(passed),failure=reason,tail_mean_abs_roll_deg=mean_abs,
                final_roll_deg=final_roll,final_rate=final_rate,neutral_offset=offset,
                roll_inertia=roll_inertia,actuator_gear=gear,trace=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--robust',action='store_true');args=parser.parse_args()
    out=Path(('results/roll_closed_loop_pilot' if args.pilot else 'results/roll_closed_loop')+('_robust' if args.robust else ''))
    out.mkdir(parents=True,exist_ok=True)
    candidate=select_model(args.robust)
    (out/'decoder.json').write_text(json.dumps(candidate,indent=2))
    print('Frozen decoder:',candidate['kind'],candidate['penalty'],flush=True)
    brain=RawMotors(verbose=False);vehicle=build_vehicle()
    scenarios=SCENARIOS[:1] if args.pilot else SCENARIOS
    summaries=[]
    for angle,rate,seed in scenarios:
        for mode in ('oracle','none','neural','disconnected'):
            path=out/f'{seed}_{mode}.json'
            if path.exists():result=json.loads(path.read_text())
            else:
                result=experiment(brain,candidate,vehicle,mode,angle,rate,seed)
                path.write_text(json.dumps(result,allow_nan=False))
            summary={k:v for k,v in result.items() if k!='trace'}
            summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False))


if __name__=='__main__':main()
