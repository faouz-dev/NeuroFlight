import os

import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback
)
from stable_baselines3.common.env_checker import (
    check_env
)
from stable_baselines3.common.monitor import (
    Monitor
)

from env.neuroflight_v2_env import (
    NeuroFlightV2Env
)

from archive.ppo_prototypes.policy_v2 import (
    FlyBrainV2Extractor
)


# ============================================================
# FICHIERS
# ============================================================

NODES = (
    "data/male_cns/neuroflight/"
    "flight_subgraph_nodes.feather"
)

EDGES = (
    "data/male_cns/neuroflight/"
    "flight_subgraph_edges.feather"
)


os.makedirs(
    "models",
    exist_ok=True,
)

os.makedirs(
    "models/checkpoints_v2",
    exist_ok=True,
)


# ============================================================
# TEST ENVIRONNEMENT
# ============================================================

print(
    "Verification environnement V2..."
)


test_env = (
    NeuroFlightV2Env(
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    )
)


check_env(
    test_env,
    warn=True,
)

test_env.close()


print(
    "Environnement V2 valide."
)


# ============================================================
# TRAIN ENV
# ============================================================

env = Monitor(
    NeuroFlightV2Env(
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    )
)


# ============================================================
# DEVICE
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device :",
    device
)


# ============================================================
# FLYBRAIN POLICY
# ============================================================

policy_kwargs = dict(

    features_extractor_class=(
        FlyBrainV2Extractor
    ),

    features_extractor_kwargs=dict(

        nodes_file=NODES,

        edges_file=EDGES,

        brain_steps=12,
    ),


    # Actor :
    #
    # 772 motor neurons
    # ->
    # directement 4 commandes.
    #
    # Pas de MLP acteur intermédiaire.

    net_arch=dict(
        pi=[],
        vf=[64, 64],
    ),


    activation_fn=torch.nn.Tanh,


    # V1 avait std ≈ 1
    # donc énormément d'actions violentes.
    #
    # Ici :
    # exp(-1) ≈ 0.37

    log_std_init=-1.0,
)


# ============================================================
# PPO
# ============================================================

model = PPO(

    policy="MlpPolicy",

    env=env,

    policy_kwargs=policy_kwargs,

    learning_rate=2e-4,

    n_steps=1024,

    batch_size=64,

    # Moins de passes que les 10 par défaut.
    # Le FlyBrain coûte cher.
    n_epochs=5,

    gamma=0.99,

    gae_lambda=0.95,

    clip_range=0.2,

    ent_coef=0.002,

    verbose=1,

    tensorboard_log="./logs/",

    device=device,
)


# ============================================================
# CHECKPOINT
# ============================================================

checkpoint_callback = (
    CheckpointCallback(

        save_freq=10_000,

        save_path=(
            "models/checkpoints_v2"
        ),

        name_prefix=(
            "neuroflight_malecns_v2"
        ),
    )
)


# ============================================================
# TRAIN
# ============================================================

print()
print(
    "=" * 60
)

print(
    " NEUROFLIGHT - MALE CNS V2"
)

print(
    "=" * 60
)

print()

print(
    "HALTERE : gyro + acceleration"
)

print(
    "VS / HS : optic flow"
)

print(
    "Connectome MaleCNS : fixe"
)

print(
    "Objectif : rester en vol"
)

print()


model.learn(

    total_timesteps=50_000,

    progress_bar=True,

    callback=checkpoint_callback,
)


# ============================================================
# SAVE
# ============================================================

model.save(
    "models/neuroflight_malecns_v2"
)


env.close()


print()
print(
    "V2 sauvegardee : "
    "models/neuroflight_malecns_v2.zip"
)