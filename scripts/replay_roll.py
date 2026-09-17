"""Animated MuJoCo replay of saved roll angles, without loading neural data.

Only roll was logged for visualization: translation is centered and pitch/yaw
are zero. This replay is not a new experiment or a full-state trajectory replay.
"""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle', type=int, choices=(-12, 12), default=12)
    parser.add_argument('--seed', type=int, choices=(1100001, 1200001), default=1100001)
    parser.add_argument('--variant', choices=('baseline', 'reduced'), default='reduced')
    parser.add_argument('--speed', type=float, default=1., help='Playback speed; 0.5 is half speed')
    parser.add_argument('--check', action='store_true', help='Validate all poses without opening a window')
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    import mujoco
    from scripts.demo_neuroflight_3d_v2 import build_vehicle, euler_to_quat_wxyz
    root = Path(__file__).resolve().parents[1]
    path = root/'results/roll_gain_comparison'/f'{args.seed}_{args.angle:+d}_{args.variant}.json'
    result = json.loads(path.read_text())
    samples = [dict(t=0., roll_deg=float(args.angle))] + result['trace']
    if not all(math.isfinite(x['t']) and math.isfinite(x['roll_deg']) for x in samples):
        raise ValueError('Non-finite trace')
    if any(a['t'] >= b['t'] for a,b in zip(samples,samples[1:])):
        raise ValueError('Trace timestamps must increase')
    model, data, *_ = build_vehicle()

    def pose(sample):
        data.qpos[:3] = [0, 0, .3]
        data.qpos[3:7] = euler_to_quat_wxyz(math.radians(sample['roll_deg']), 0., 0.)
        data.time = sample['t']
        mujoco.mj_forward(model, data)

    print(f'RELECTURE — {path.name}', flush=True)
    print('Angles enregistrés ; position recentrée ; aucun calcul neuronal relancé.', flush=True)
    print(f"Critère {'atteint' if result['passed'] else 'NON atteint'} ; "
          f"erreur moyenne en fin d'essai : {result['tail_mean_abs_roll_deg']:.2f} degrés.", flush=True)
    if args.check:
        for sample in samples:
            pose(sample)
        print(f'{len(samples)} poses vérifiées sans fenêtre.')
        return
    import mujoco.viewer
    pose(samples[0])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, .3]
        viewer.cam.distance = .32
        viewer.cam.azimuth = 0
        viewer.cam.elevation = -15
        print('Lecture en boucle. Fermer la fenêtre pour quitter.', flush=True)
        while viewer.is_running():
            start = time.perf_counter()
            for sample in samples:
                deadline = start + sample['t']/args.speed
                while viewer.is_running() and time.perf_counter() < deadline:
                    time.sleep(min(.01, max(0., deadline-time.perf_counter())))
                if not viewer.is_running():
                    break
                with viewer.lock():
                    pose(sample)
                viewer.sync()


if __name__ == '__main__':
    main()
