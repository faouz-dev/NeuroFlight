from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class FlyBrainNetwork(nn.Module):

    def __init__(
        self,
        nodes_file,
        edges_file,
        device="cpu",
    ):
        super().__init__()

        self.device = torch.device(device)

        print("Chargement du sous-reseau MaleCNS...")

        # ====================================================
        # DONNEES
        # ====================================================

        self.nodes = pd.read_feather(nodes_file)
        self.edges = pd.read_feather(edges_file)

        self.nodes = (
            self.nodes
            .drop_duplicates(subset=["bodyId"])
            .reset_index(drop=True)
        )

        print("Neurones    :", len(self.nodes))
        print("Connexions  :", len(self.edges))


        # ====================================================
        # BODY ID -> INDEX INTERNE
        # ====================================================

        self.body_to_index = {
            int(body_id): i
            for i, body_id
            in enumerate(self.nodes["bodyId"])
        }

        self.index_to_body = {
            i: int(body_id)
            for body_id, i
            in self.body_to_index.items()
        }

        self.num_neurons = len(self.nodes)


        # ====================================================
        # ROLES
        # ====================================================

        roles = (
            self.nodes["neuroflight_role"]
            .fillna("")
            .astype(str)
        )

        sensory_indices = np.where(
            roles == "sensory"
        )[0]

        descending_indices = np.where(
            roles == "descending"
        )[0]

        motor_indices = np.where(
            roles == "motor"
        )[0]


        self.register_buffer(
            "sensory_indices",
            torch.tensor(
                sensory_indices,
                dtype=torch.long,
            )
        )

        self.register_buffer(
            "descending_indices",
            torch.tensor(
                descending_indices,
                dtype=torch.long,
            )
        )

        self.register_buffer(
            "motor_indices",
            torch.tensor(
                motor_indices,
                dtype=torch.long,
            )
        )


        print(
            "Sensoriels   :",
            len(sensory_indices)
        )

        print(
            "Descending   :",
            len(descending_indices)
        )

        print(
            "Moteurs      :",
            len(motor_indices)
        )


        # ====================================================
        # NEUROTRANSMETTEURS
        #
        # Approximation initiale.
        #
        # IMPORTANT :
        # la topologie vient du MaleCNS.
        #
        # Le signe fonctionnel ci-dessous est une
        # approximation pour notre premier simulateur.
        # ====================================================

        nt_by_body = (
            self.nodes
            .set_index("bodyId")["consensus_nt"]
            .fillna("unknown")
            .astype(str)
            .str.lower()
            .to_dict()
        )


        def neurotransmitter_sign(body_id):

            nt = nt_by_body.get(
                int(body_id),
                "unknown",
            )

            if nt == "acetylcholine":
                return 1.0

            if nt == "gaba":
                return -1.0

            # Approximation initiale pour le glutamate
            if nt == "glutamate":
                return -1.0

            # Sérotonine, dopamine, etc. sont
            # réellement modulatoires.
            #
            # Pour cette première simulation,
            # on leur donne un effet faible.
            return 0.25


        # ====================================================
        # CONSTRUCTION DE LA MATRICE CREUSE
        #
        # W[post, pre]
        #
        # activité suivante =
        # W @ activité actuelle
        # ====================================================

        pre_indices = []
        post_indices = []
        raw_weights = []


        for row in self.edges.itertuples(
            index=False
        ):

            pre_body = int(row.body_pre)
            post_body = int(row.body_post)

            if (
                pre_body not in self.body_to_index
                or
                post_body not in self.body_to_index
            ):
                continue

            pre = self.body_to_index[
                pre_body
            ]

            post = self.body_to_index[
                post_body
            ]

            # Nombre de synapses -> force
            #
            # log1p évite qu'une connexion de
            # plusieurs milliers de synapses
            # domine complètement le réseau.

            magnitude = np.log1p(
                float(row.weight)
            )

            sign = neurotransmitter_sign(
                pre_body
            )

            weight = (
                magnitude * sign
            )


            pre_indices.append(pre)
            post_indices.append(post)
            raw_weights.append(weight)


        pre_indices = np.asarray(
            pre_indices,
            dtype=np.int64,
        )

        post_indices = np.asarray(
            post_indices,
            dtype=np.int64,
        )

        raw_weights = np.asarray(
            raw_weights,
            dtype=np.float32,
        )


        # ====================================================
        # NORMALISATION PAR NEURONE POSTSYNAPTIQUE
        #
        # Cela évite une explosion immédiate de l'activité.
        # ====================================================

        incoming_strength = np.bincount(
            post_indices,
            weights=np.abs(raw_weights),
            minlength=self.num_neurons,
        ).astype(np.float32)


        incoming_strength[
            incoming_strength < 1.0
        ] = 1.0


        normalized_weights = (
            raw_weights
            /
            incoming_strength[
                post_indices
            ]
        )


        # Gain global initial
        normalized_weights *= 1.25


        # ====================================================
        # TENSEUR SPARSE PYTORCH
        # ====================================================

        indices = torch.tensor(
            np.vstack(
                [
                    post_indices,
                    pre_indices,
                ]
            ),
            dtype=torch.long,
        )

        values = torch.tensor(
            normalized_weights,
            dtype=torch.float32,
        )

        W = torch.sparse_coo_tensor(
            indices,
            values,
            size=(
                self.num_neurons,
                self.num_neurons,
            ),
        ).coalesce()


        self.register_buffer(
            "W",
            W,
        )


        # ====================================================
        # PARAMETRES DYNAMIQUES
        #
        # Ce ne sont PAS des données du connectome.
        #
        # Ils décrivent notre modèle dynamique provisoire.
        # ====================================================

        self.alpha = 0.35
        self.decay = 0.92


        self.to(self.device)


        print()
        print(
            "FlyBrainNetwork pret."
        )


    # ========================================================
    # CREER ETAT VIDE
    # ========================================================

    def initial_state(self):

        return torch.zeros(
            self.num_neurons,
            dtype=torch.float32,
            device=self.device,
        )


    # ========================================================
    # UNE ETAPE NEURONALE
    # ========================================================

    def step(
        self,
        state,
        sensory_drive,
    ):

        # Activité venant du connectome

        synaptic_input = torch.sparse.mm(
            self.W,
            state.unsqueeze(1),
        ).squeeze(1)


        # Entrée externe venant des capteurs

        external = torch.zeros_like(
            state
        )

        external[
            self.sensory_indices
        ] = sensory_drive


        total_input = (
            synaptic_input
            + external
        )


        # Modèle neuronal rate-based
        candidate = torch.tanh(
            total_input
        )


        # Petite mémoire / fuite temporelle

        new_state = (
            self.decay
            * (1.0 - self.alpha)
            * state
            +
            self.alpha
            * candidate
        )


        return new_state


    # ========================================================
    # PLUSIEURS ETAPES
    # ========================================================

    def run(
        self,
        sensory_drive,
        steps=20,
        state=None,
    ):

        if state is None:
            state = self.initial_state()


        for _ in range(steps):

            state = self.step(
                state,
                sensory_drive,
            )


        return state


    # ========================================================
    # ACTIVITE DES DN
    # ========================================================

    def descending_activity(
        self,
        state,
    ):

        return state[
            self.descending_indices
        ]


    # ========================================================
    # ACTIVITE DES MOTOR NEURONS
    # ========================================================

    def motor_activity(
        self,
        state,
    ):

        return state[
            self.motor_indices
        ]
        
    def run_batch(
    self,
    sensory_drive,
    steps=12,
    state=None,
    ):

    # sensory_drive :
    # [batch, nombre_neurones_sensoriels]

        if sensory_drive.dim() == 1:
            sensory_drive = sensory_drive.unsqueeze(0)

        batch_size = sensory_drive.shape[0]

        if state is None:
            state = torch.zeros(
                batch_size,
                self.num_neurons,
                dtype=sensory_drive.dtype,
                device=sensory_drive.device,
            )

        for _ in range(steps):

            # W @ activité pour chaque élément du batch
            synaptic_input = torch.sparse.mm(
                self.W,
                state.transpose(0, 1),
            ).transpose(0, 1)

            external = torch.zeros_like(state)

            external[
                :,
                self.sensory_indices
            ] = sensory_drive

            candidate = torch.tanh(
                synaptic_input + external
            )

            state = (
                self.decay
                * (1.0 - self.alpha)
                * state
                +
                self.alpha
                * candidate
            )

        return state