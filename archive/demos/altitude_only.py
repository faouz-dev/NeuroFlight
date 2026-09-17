import time
import mujoco
import mujoco.viewer
import mujoco_menagerie as mm

# Charger le Crazyflie
model = mm.load("bitcraze_crazyflie_2")
data = mujoco.MjData(model)

# Masse et gravité
mass = model.body_mass.sum()
gravity = abs(model.opt.gravity[2])

# Poussée qui compense exactement la gravité
hover_thrust = mass * gravity

print("Masse :", mass, "kg")
print("Poussee stationnaire :", hover_thrust, "N")

# Position initiale : 10 cm
data.qpos[2] = 0.10
mujoco.mj_forward(model, data)

# Altitude voulue : 50 cm
target_z = 0.50

# Petit contrôleur vertical PD
KP = 0.15
KD = 0.08

last_print = 0

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        step_start = time.time()

        # Altitude actuelle
        z = data.qpos[2]

        # Vitesse verticale
        vz = data.qvel[2]

        # Erreur d'altitude
        error = target_z - z

        # Calcul de la poussée
        thrust = (
            hover_thrust
            + KP * error
            - KD * vz
        )

        # Limite physique du modèle
        thrust = max(0.0, min(0.35, thrust))

        # Commandes
        data.ctrl[0] = thrust

        # Pas encore de contrôle de rotation
        data.ctrl[1] = 0.0
        data.ctrl[2] = 0.0
        data.ctrl[3] = 0.0

        # Simulation physique
        mujoco.mj_step(model, data)

        # Mettre à jour l'affichage
        viewer.sync()

        # Afficher l'altitude environ 2 fois/seconde
        if data.time - last_print > 0.5:
            print(
                f"Altitude: {z:.3f} m | "
                f"Vitesse: {vz:.3f} m/s | "
                f"Poussee: {thrust:.3f} N"
            )
            last_print = data.time

        # Temps réel approximatif
        remaining = model.opt.timestep - (time.time() - step_start)

        if remaining > 0:
            time.sleep(remaining)