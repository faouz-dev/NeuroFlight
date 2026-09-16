import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from archive.ppo_prototypes.neuroflight_env import NeuroFlightEnv


# ============================================================
# DOSSIER MODELES
# ============================================================

os.makedirs("models", exist_ok=True)


# ============================================================
# ENVIRONNEMENT
# ============================================================

env = NeuroFlightEnv(
    target_z=0.50,
    frame_skip=5,
    max_steps=1000,
)

# Vérification Gymnasium / Stable-Baselines3
print("Verification de l'environnement...")

check_env(
    env,
    warn=True,
)

env.close()

print("Environnement valide.")


# ============================================================
# ENVIRONNEMENT D'ENTRAINEMENT
# ============================================================

env = Monitor(
    NeuroFlightEnv(
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    )
)


# ============================================================
# RESEAU + PPO
# ============================================================

model = PPO(
    policy="MlpPolicy",
    env=env,

    learning_rate=3e-4,

    n_steps=2048,
    batch_size=64,

    gamma=0.99,
    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.01,

    verbose=1,

    tensorboard_log="./logs/",
)


# ============================================================
# ENTRAINEMENT
# ============================================================

print()
print("=====================================")
print(" DEBUT ENTRAINEMENT NEUROFLIGHT")
print("=====================================")
print()

model.learn(
    total_timesteps=300_000,
    progress_bar=True,
)


# ============================================================
# SAUVEGARDE
# ============================================================

model.save(
    "models/neuroflight_ppo"
)

env.close()

print()
print("Entrainement termine.")
print(
    "Modele sauvegarde : "
    "models/neuroflight_ppo.zip"
)