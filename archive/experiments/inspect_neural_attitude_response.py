"""Raw MaleCNS motor-neuron response to static drone attitude.

No MuJoCo moments, no neural decoder and no fly-to-drone translation are used.
The output is only the four directly observed motor populations:
 b12_L, b12_R, b3i1_L, b3i1_R.
"""
from __future__ import annotations
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from scripts.demo_neuroflight_3d_v2 import NeuroFlight3AxisControllerV2, GYRO_TO_HZ, DELTA_RATE_HZ

ANGLES_DEG=(-12.,-6.,0.,6.,12.)
TRIALS=4
MEASURE_MS=500.
SAMPLE_EVERY_MS=5.
ATTITUDE_K=np.array([1.10,.85,.60])
OUT=Path('logs/neural_attitude_raw')/datetime.now().strftime('%Y%m%d_%H%M%S'); OUT.mkdir(parents=True,exist_ok=True)

def one_trial(controller, euler_rad, seed):
    controller.reset_runtime(seed=seed)
    # This is the same sensory input model used in flight, but readout remains raw.
    requested=np.clip((GYRO_TO_HZ*(ATTITUDE_K*euler_rad))/DELTA_RATE_HZ,-1.,1.)
    steps=int(round(MEASURE_MS/controller.brain.dt_ms)); stride=max(1,int(round(SAMPLE_EVERY_MS/controller.brain.dt_ms)))
    values=[]
    for i in range(steps):
        controller._tick(controller.runtime_rng,requested)  # raw LIF motor traces only
        if i%stride==0: values.append(controller.traces.copy())
    return np.mean(values,axis=0),requested

def main():
    brain=NeuroFlight3AxisControllerV2(seed=4242,verbose=True)
    names=('b12_L','b12_R','b3i1_L','b3i1_R'); rows=[]; seed=9000
    print('\nRAW NEURAL ATTITUDE RESPONSE — no decoded command, no drone moments')
    for axis,axis_name in enumerate(('roll','pitch','yaw')):
        print(f'\n{axis_name.upper()} static tilt')
        for angle in ANGLES_DEG:
            euler=np.zeros(3); euler[axis]=np.radians(angle)
            trials=[]
            for _ in range(TRIALS):
                result,requested=one_trial(brain,euler,seed); seed+=1; trials.append(result)
            arr=np.asarray(trials); mean=arr.mean(axis=0); std=arr.std(axis=0)
            row={'axis':axis_name,'angle_deg':angle,'requested_roll':requested[0],'requested_pitch':requested[1],'requested_yaw':requested[2]}
            row.update({name:mean[i] for i,name in enumerate(names)}); row.update({f'std_{name}':std[i] for i,name in enumerate(names)})
            rows.append(row)
            print(f'{angle:+5.1f} deg | '+ ' | '.join(f'{name}={mean[i]:7.3f}±{std[i]:.3f}' for i,name in enumerate(names)))
    with (OUT/'raw_motor_response.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary='''RAW NEURAL ATTITUDE RESPONSE\n\nThis experiment applies static attitude only to the haltere sensory encoder.\nNo decoded roll/pitch/yaw command, no moment actuator and no vehicle flight are used.\n\nRead raw_motor_response.csv: for each axis, compare -12, -6, 0, +6, +12 degrees.\nA useful raw neural response changes progressively with the angle and reverses between negative and positive tilt.\n'''
    (OUT/'SUMMARY.txt').write_text(summary,encoding='utf-8'); print(f'\nFiles: {OUT}')
if __name__=='__main__': main()
