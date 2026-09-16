import os
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from archive.ppo_prototypes.neuroflight_env import NeuroFlightEnv
from archive.ppo_prototypes.policy import FlyBrainExtractor


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


# ============================================================
# ENVIRONNEMENT
# ============================================================

env = Monitor(
    NeuroFlightEnv(
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    )
)


# ============================================================
# POLITIQUE FLYBRAIN
# ============================================================

policy_kwargs = dict(

    features_extractor_class=(
        FlyBrainExtractor
    ),

    features_extractor_kwargs=dict(
        nodes_file=NODES,
        edges_file=EDGES,

        # Nombre d'iterations neuronales
        # entre entrée sensorielle et sortie.
        brain_steps=12,
    ),


    # IMPORTANT :
    #
    # AUCUN MLP supplémentaire
    # pour l'acteur.
    #
    # Les 772 motor neurons vont directement
    # vers les 4 commandes via une couche linéaire.
    #
    # Le réseau 64/64 ci-dessous est uniquement
    # utilisé par le critic PPO pour estimer
    # la valeur, pas pour piloter le drone.

    net_arch=dict(
        pi=[],
        vf=[64, 64],
    ),

    activation_fn=torch.nn.Tanh,
)


model = PPO(

    policy="MlpPolicy",

    env=env,

    policy_kwargs=policy_kwargs,

    learning_rate=1e-4,

    n_steps=1024,

    batch_size=64,

    gamma=0.99,

    gae_lambda=0.95,

    ent_coef=0.005,

    verbose=1,

    tensorboard_log="./logs/",
)


print()
print("=" * 60)
print(" NEUROFLIGHT - MALE CNS TRAINING")
print("=" * 60)
print()

print(
    "Le connectome MaleCNS reste fixe."
)

print(
    "Entrainement des adaptateurs "
    "sensoriels et moteurs."
)

print()


# Premier essai court
model.learn(
    total_timesteps=100_000,
    progress_bar=True,
)


model.save(
    "models/neuroflight_malecns_v1"
)


env.close()


print()
print(
    "Modele sauvegarde : "
    "models/neuroflight_malecns_v1.zip"
)