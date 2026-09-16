import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)

from archive.ppo_prototypes.network import (
    FlyBrainNetwork
)


class FlyBrainV2Extractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        nodes_file,
        edges_file,
        brain_steps=12,
    ):

        nodes = pd.read_feather(
            nodes_file
        )

        motor_count = int(
            (
                nodes[
                    "neuroflight_role"
                ]
                == "motor"
            ).sum()
        )


        super().__init__(
            observation_space,
            features_dim=motor_count,
        )


        self.brain_steps = (
            brain_steps
        )


        # ====================================================
        # MALE CNS
        # ====================================================

        self.brain = FlyBrainNetwork(
            nodes_file=nodes_file,
            edges_file=edges_file,
        )


        sensory_global = (
            self.brain
            .sensory_indices
            .cpu()
            .numpy()
        )


        sensory_nodes = (
            self.brain.nodes
            .iloc[sensory_global]
            .reset_index(drop=True)
        )


        # ====================================================
        # HALTERE
        # ====================================================

        haltere_mask = (
            sensory_nodes[
                "subclass"
            ]
            .fillna("")
            .astype(str)
            .str.lower()
            .eq("haltere")
            .to_numpy()
        )


        # ====================================================
        # VISUAL VS / HS
        # ====================================================

        visual_types = {
            "VS",
            "HSN",
            "HSE",
            "HSS",
        }


        visual_mask = (
            sensory_nodes[
                "type"
            ]
            .fillna("")
            .astype(str)
            .isin(
                visual_types
            )
            .to_numpy()
        )


        haltere_positions = (
            np.where(
                haltere_mask
            )[0]
        )

        visual_positions = (
            np.where(
                visual_mask
            )[0]
        )


        self.register_buffer(
            "haltere_positions",
            torch.tensor(
                haltere_positions,
                dtype=torch.long,
            )
        )


        self.register_buffer(
            "visual_positions",
            torch.tensor(
                visual_positions,
                dtype=torch.long,
            )
        )


        self.num_sensory = len(
            sensory_global
        )

        self.num_haltere = len(
            haltere_positions
        )

        self.num_visual = len(
            visual_positions
        )


        print()
        print(
            "=============================="
        )

        print(
            "NEUROFLIGHT FLYBRAIN V2"
        )

        print(
            "=============================="
        )

        print(
            "Sensoriels total :",
            self.num_sensory
        )

        print(
            "Haltere :",
            self.num_haltere
        )

        print(
            "VS / HS :",
            self.num_visual
        )

        print(
            "Motor neurons :",
            motor_count
        )


        # ====================================================
        # ADAPTATEUR HALTERE
        #
        # current :
        # gyro 3 + accel 3
        #
        # delta :
        # gyro 3 + accel 3
        #
        # TOTAL = 12
        # ====================================================

        self.haltere_adapter = (
            nn.Linear(
                12,
                self.num_haltere,
            )
        )


        # ====================================================
        # ADAPTATEUR VISUEL
        #
        # flow translation 3
        # flow rotation 3
        #
        # +
        #
        # variations correspondantes
        #
        # TOTAL = 12
        # ====================================================

        self.visual_adapter = (
            nn.Linear(
                12,
                self.num_visual,
            )
        )


        # Initialisation calme
        nn.init.normal_(
            self.haltere_adapter.weight,
            mean=0.0,
            std=0.03,
        )

        nn.init.zeros_(
            self.haltere_adapter.bias
        )


        nn.init.normal_(
            self.visual_adapter.weight,
            mean=0.0,
            std=0.03,
        )

        nn.init.zeros_(
            self.visual_adapter.bias
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        observations,
    ):

        # observations :
        #
        # 0:12  signaux actuels
        # 12:24 delta temporel


        current = observations[
            :,
            0:12
        ]

        delta = observations[
            :,
            12:24
        ]


        # ----------------------------------------------------
        # HALTERE
        # ----------------------------------------------------

        haltere_features = torch.cat(
            [
                current[:, 0:6],
                delta[:, 0:6],
            ],
            dim=1,
        )


        haltere_activity = torch.tanh(
            self.haltere_adapter(
                haltere_features
            )
        )


        # ----------------------------------------------------
        # VISUAL
        # ----------------------------------------------------

        visual_features = torch.cat(
            [
                current[:, 6:12],
                delta[:, 6:12],
            ],
            dim=1,
        )


        visual_activity = torch.tanh(
            self.visual_adapter(
                visual_features
            )
        )


        # ----------------------------------------------------
        # ACTIVITE SENSORIELLE COMPLETE
        # ----------------------------------------------------

        sensory_drive = torch.zeros(
            observations.shape[0],
            self.num_sensory,
            dtype=observations.dtype,
            device=observations.device,
        )


        sensory_drive = (
            sensory_drive.index_copy(
                1,
                self.haltere_positions,
                haltere_activity,
            )
        )


        sensory_drive = (
            sensory_drive.index_copy(
                1,
                self.visual_positions,
                visual_activity,
            )
        )


        # ----------------------------------------------------
        # CONNECTOME
        # ----------------------------------------------------

        state = (
            self.brain.run_batch(
                sensory_drive,
                steps=self.brain_steps,
            )
        )


        # ----------------------------------------------------
        # SORTIE MOTOR NEURONS
        # ----------------------------------------------------

        motor_activity = state[
            :,
            self.brain.motor_indices
        ]


        return motor_activity