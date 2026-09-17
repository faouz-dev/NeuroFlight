"""Generate figures from committed traces; no neural simulation is rerun.

MuJoCo views reconstruct logged roll only. Translation is deliberately centered;
they are not screenshots of a complete saved simulator state.
"""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'results/roll_gain_comparison'
OUT = ROOT/'docs/assets'


def read(seed, angle, variant):
    return json.loads((RESULTS/f'{seed}_{angle:+g}_{variant}.json').read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, (seed, angle) in zip(axes.flat, [(1100001,12),(1100001,-12),(1200001,12),(1200001,-12)]):
        ax.axhspan(-5,5,color='#e7f3ee',label='Repère ±5° (critère complet : README)')
        for variant, label, color in [('baseline','Réglage initial','#7b8795'),('reduced','Retour plus doux','#007d86')]:
            r=read(seed,angle,variant)
            ax.plot([x['t'] for x in r['trace']], [x['roll_deg'] for x in r['trace']], label=label, color=color, lw=2)
        ax.axhline(0,color='#405064',lw=.7)
        ax.set_title(f"{'Diagnostic' if seed==1100001 else 'Confirmation'} · départ {angle:+}°")
        ax.set(xlim=(0,4),ylim=(-22,22),xlabel='Temps (s)',ylabel='Roulis (°)')
        ax.grid(alpha=.2)
    handles, labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.suptitle('NeuroFlight — 8 essais, 2 tirages aléatoires, un seul axe',fontsize=16)
    fig.tight_layout(rect=(0,.06,1,.95))
    fig.savefig(OUT/'roll_comparison.png',dpi=140)
    plt.close(fig)
    if args.plots_only:
        return
    import mujoco
    from scripts.demo_neuroflight_3d_v2 import build_vehicle, euler_to_quat_wxyz
    m,d,*_=build_vehicle()
    m.vis.global_.offwidth=480
    m.vis.global_.offheight=320
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0,0,.3]
    camera.distance=.32
    camera.azimuth=0
    camera.elevation=-15
    fig,axes=plt.subplots(2,3,figsize=(12,7.2))
    with mujoco.Renderer(m,height=320,width=480) as renderer:
        for row,angle in enumerate((12,-12)):
            trial=read(1100001,angle,'reduced')
            for col,t in enumerate((0.,2.,4.)):
                if t==0.:
                    measured=angle;stamp=0.
                else:
                    sample=min(trial['trace'],key=lambda x:abs(x['t']-t))
                    measured=sample['roll_deg'];stamp=sample['t']
                mujoco.mj_resetData(m,d)
                d.qpos[:3]=[0,0,.3]
                d.qpos[3:7]=euler_to_quat_wxyz(math.radians(measured),0,0)
                mujoco.mj_forward(m,d)
                renderer.update_scene(d,camera=camera)
                axes[row,col].imshow(renderer.render().copy())
                axes[row,col].set_title(f't = {stamp:.2f} s · roulis {measured:+.1f}°')
                axes[row,col].axis('off')
    fig.suptitle('Rendu MuJoCo des angles enregistrés — retour plus doux',fontsize=15)
    fig.text(.5,.025,'Haut : critère atteint · Bas : critère non atteint · Position recentrée, roulis seul reconstruit',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.94),h_pad=2.8)
    fig.savefig(OUT/'drone_roll_replay.png',dpi=130)
    plt.close(fig)


if __name__=='__main__':
    main()
