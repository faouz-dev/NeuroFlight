import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)

from archive.ppo_prototypes.network import FlyBrainNetwork


class FlyBrainExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space,
        nodes_file,
        edges_file,
        brain_steps=12,
    ):

        # Lire uniquement pour connaitre
        # le nombre de motor neurons
        nodes = pd.read_feather(
            nodes_file
        )

        motor_count = int(
            (
                nodes["neuroflight_role"]
                == "motor"
            ).sum()
        )

        super().__init__(
            observation_space,
            features_dim=motor_count,
        )

        self.brain_steps = brain_steps


        # ====================================================
        # CONNECTOME MALE CNS
        # ====================================================

        self.brain = FlyBrainNetwork(
            nodes_file=nodes_file,
            edges_file=edges_file,
        )


        # ====================================================
        # TROUVER LES NEURONES HALTERE
        # ====================================================

        sensory_global_indices = (
            self.brain
            .sensory_indices
            .cpu()
            .numpy()
        )

        sensory_nodes = (
            self.brain.nodes
            .iloc[sensory_global_indices]
            .reset_index(drop=True)
        )

        haltere_mask = (
            sensory_nodes["subclass"]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("haltere")
            .to_numpy()
        )

        haltere_positions = np.where(
            haltere_mask
        )[0]


        self.register_buffer(
            "haltere_positions",
            torch.tensor(
                haltere_positions,
                dtype=torch.long,
            )
        )


        self.num_sensory = len(
            sensory_global_indices
        )

        self.num_haltere = len(
            haltere_positions
        )


        print()
        print(
            "FlyBrain PPO"
        )

        print(
            "Neurones sensoriels :",
            self.num_sensory
        )

        print(
            "Neurones haltere :",
            self.num_haltere
        )

        print(
            "Neurones moteurs :",
            motor_count
        )


        # ====================================================
        # ADAPTATEUR CAPTEUR
        #
        # 6 entrées :
        #
        # gyro X/Y/Z
        # acceleration X/Y/Z
        #
        # ->
        #
        # neurones haltere
        #
        # C'est entraînable.
        # ====================================================

        self.sensor_adapter = nn.Linear(
            6,
            self.num_haltere,
        )


    def forward(self, observations):

        # ====================================================
        # OBSERVATION NEUROFLIGHT
        #
        # 0:3   position
        # 3:6   vitesse
        # 6:10  quaternion
        # 10:13 gyro
        # 13:16 acceleration
        # ====================================================

        gyro = observations[
            :,
            10:13
        ]

        acceleration = observations[
            :,
            13:16
        ]


        haltere_input = torch.cat(
            [
                gyro,
                acceleration,
            ],
            dim=1,
        )


        # ====================================================
        # DRONE -> HALTERES
        # ====================================================

        haltere_activity = torch.tanh(
            self.sensor_adapter(
                haltere_input
            )
        )


        sensory_drive = torch.zeros(
            observations.shape[0],
            self.num_sensory,
            dtype=observations.dtype,
            device=observations.device,
        )


        sensory_drive[
            :,
            self.haltere_positions
        ] = haltere_activity


        # ====================================================
        # MALE CNS
        # ====================================================

        state = self.brain.run_batch(
            sensory_drive,
            steps=self.brain_steps,
        )


        # ====================================================
        # SORTIE :
        # 772 MOTOR NEURONS
        # ====================================================

        motor_activity = state[
            :,
            self.brain.motor_indices
        ]


        return motor_activity