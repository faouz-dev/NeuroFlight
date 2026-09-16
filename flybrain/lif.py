from pathlib import Path
import json

import numpy as np


class MaleCNSLIF:

    def __init__(
        self,
        cache_dir="data/male_cns/cache_v3",
        dt_ms=0.5,

        # Paramètres Shiu et al.
        v_rest=-52.0,
        v_reset=-52.0,
        v_threshold=-45.0,

        tau_membrane_ms=20.0,
        tau_synapse_ms=5.0,

        refractory_ms=2.2,
        delay_ms=1.8,

        weight_per_synapse_mv=0.275,

        # 1.0 = poids nominal Shiu.
        #
        # 0.65 est une calibration pratique utilisée
        # par un port MaleCNS pour éviter une activité
        # excessive sur ce graphe plus dense.
        global_gain=0.65,
    ):

        self.cache_dir = Path(
            cache_dir
        )

        self.dt_ms = float(
            dt_ms
        )

        self.v_rest = float(
            v_rest
        )

        self.v_reset = float(
            v_reset
        )

        self.v_threshold = float(
            v_threshold
        )

        self.tau_membrane_ms = float(
            tau_membrane_ms
        )

        self.tau_synapse_ms = float(
            tau_synapse_ms
        )

        self.refractory_period_ms = float(
            refractory_ms
        )

        self.delay_ms = float(
            delay_ms
        )

        self.weight_per_synapse_mv = float(
            weight_per_synapse_mv
        )

        self.global_gain = float(
            global_gain
        )


        # ====================================================
        # CACHE
        # ====================================================

        print(
            "Chargement MaleCNS LIF..."
        )


        self.body_ids = np.load(
            self.cache_dir
            / "body_ids.npy",
            mmap_mode="r",
        )

        self.row_ptr = np.load(
            self.cache_dir
            / "row_ptr.npy",
            mmap_mode="r",
        )

        self.post_idx = np.load(
            self.cache_dir
            / "post_idx.npy",
            mmap_mode="r",
        )

        self.syn_count = np.load(
            self.cache_dir
            / "syn_count.npy",
            mmap_mode="r",
        )

        self.nt_code = np.load(
            self.cache_dir
            / "nt_code.npy",
            mmap_mode="r",
        )


        self.num_neurons = len(
            self.body_ids
        )

        self.num_connections = len(
            self.post_idx
        )


        # ====================================================
        # NEUROTRANSMETTEURS -> SIGNE
        #
        # Convention du modèle de Shiu :
        #
        # ACh              +
        # GABA             -
        # glutamate        -
        # histamine        -
        # monoamines       +
        # unclear          +
        #
        # Ce signe reste une hypothèse du modèle dynamique,
        # pas une information directement mesurée pour chaque
        # synapse.
        # ====================================================

        sign_table = np.ones(
            9,
            dtype=np.float32,
        )

        # Codes créés par notre cache :
        #
        # 0 unclear
        # 1 ACh
        # 2 GABA
        # 3 glutamate
        # 4 histamine
        # 5 serotonin
        # 6 dopamine
        # 7 octopamine
        # 8 tyramine

        sign_table[2] = -1.0
        sign_table[3] = -1.0
        sign_table[4] = -1.0


        self.neuron_sign = (
            sign_table[
                np.asarray(
                    self.nt_code,
                    dtype=np.int8,
                )
            ]
        )


        # ====================================================
        # INTEGRATION EXACTE DES EQUATIONS LINEAIRES
        #
        # g(t+dt) = g(t) * b
        #
        # v(t+dt) =
        # Vrest
        # + (v-Vrest)*a
        # + g*c
        # ====================================================

        self.a = np.exp(
            -self.dt_ms
            / self.tau_membrane_ms
        )

        self.b = np.exp(
            -self.dt_ms
            / self.tau_synapse_ms
        )


        if abs(
            self.tau_membrane_ms
            - self.tau_synapse_ms
        ) < 1e-12:

            raise ValueError(
                "tau membrane et tau synapse "
                "doivent etre differents."
            )


        self.c = (
            self.tau_synapse_ms
            /
            (
                self.tau_membrane_ms
                - self.tau_synapse_ms
            )
            *
            (
                self.a
                - self.b
            )
        )


        # ====================================================
        # DELAI SYNAPTIQUE
        #
        # Avec dt = 0.5 ms :
        #
        # 1.8 ms ≈ 4 steps = 2.0 ms
        #
        # On conservera ce compromis pour la première
        # implémentation rapide.
        # ====================================================

        self.delay_steps = max(
            1,
            int(
                round(
                    self.delay_ms
                    / self.dt_ms
                )
            ),
        )


        self.effective_delay_ms = (
            self.delay_steps
            * self.dt_ms
        )


        self.buffer_length = (
            self.delay_steps
            + 1
        )


        # ====================================================
        # ETAT
        # ====================================================

        self.v = np.full(
            self.num_neurons,
            self.v_rest,
            dtype=np.float32,
        )

        self.g = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        self.refractory_remaining = (
            np.zeros(
                self.num_neurons,
                dtype=np.float32,
            )
        )


        self.delay_buffer = np.zeros(
            (
                self.buffer_length,
                self.num_neurons,
            ),
            dtype=np.float32,
        )


        self.step_index = 0


        print()
        print(
            "MaleCNS LIF pret."
        )

        print(
            "Neurones     :",
            self.num_neurons
        )

        print(
            "Connexions   :",
            self.num_connections
        )

        print(
            "dt           :",
            self.dt_ms,
            "ms"
        )

        print(
            "delai demande:",
            self.delay_ms,
            "ms"
        )

        print(
            "delai effectif:",
            self.effective_delay_ms,
            "ms"
        )

        print(
            "gain global  :",
            self.global_gain
        )


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.v.fill(
            self.v_rest
        )

        self.g.fill(
            0.0
        )

        self.refractory_remaining.fill(
            0.0
        )

        self.delay_buffer.fill(
            0.0
        )

        self.step_index = 0


    # ========================================================
    # BODY ID -> INDEX
    # ========================================================

    def body_to_index(
        self,
        body_id,
    ):

        position = np.searchsorted(
            self.body_ids,
            int(body_id),
        )

        if (
            position
            >= self.num_neurons
            or
            self.body_ids[position]
            != int(body_id)
        ):

            return None

        return int(position)


    # ========================================================
    # PROPAGATION DES SPIKES
    # ========================================================

    def _schedule_spikes(
        self,
        spike_indices,
    ):

        if len(spike_indices) == 0:
            return 0


        spike_indices = np.asarray(
            spike_indices,
            dtype=np.int64,
        )


        target_slot = (
            self.step_index
            + self.delay_steps
        ) % self.buffer_length


        total_edges = 0


        # Traitement par blocs pour éviter de créer
        # des tableaux temporaires gigantesques.
        CHUNK = 4096


        for start_chunk in range(
            0,
            len(spike_indices),
            CHUNK,
        ):

            neurons = spike_indices[
                start_chunk:
                start_chunk + CHUNK
            ]


            starts = np.asarray(
                self.row_ptr[
                    neurons
                ],
                dtype=np.int64,
            )

            ends = np.asarray(
                self.row_ptr[
                    neurons + 1
                ],
                dtype=np.int64,
            )


            lengths = (
                ends
                - starts
            )


            useful = (
                lengths > 0
            )


            if not useful.any():
                continue


            neurons = neurons[
                useful
            ]

            starts = starts[
                useful
            ]

            lengths = lengths[
                useful
            ]


            total = int(
                lengths.sum()
            )


            total_edges += total


            # ------------------------------------------------
            # Construire les indices des arêtes :
            #
            # [start0 ... end0,
            #  start1 ... end1,
            #  ...]
            #
            # sans boucle Python par neurone.
            # ------------------------------------------------

            group_offsets = (
                np.cumsum(
                    lengths,
                    dtype=np.int64,
                )
                - lengths
            )


            correction = np.repeat(
                starts
                - group_offsets,
                lengths,
            )


            edge_indices = (
                np.arange(
                    total,
                    dtype=np.int64,
                )
                + correction
            )


            posts = np.asarray(
                self.post_idx[
                    edge_indices
                ],
                dtype=np.int64,
            )


            synapses = np.asarray(
                self.syn_count[
                    edge_indices
                ],
                dtype=np.float32,
            )


            presynaptic_sign = np.repeat(
                self.neuron_sign[
                    neurons
                ],
                lengths,
            )


            weights = (
                synapses
                * presynaptic_sign
                * self.weight_per_synapse_mv
                * self.global_gain
            )


            # Plusieurs neurones peuvent cibler
            # le même postsynaptique.
            np.add.at(
                self.delay_buffer[
                    target_slot
                ],
                posts,
                weights,
            )


        return total_edges


    # ========================================================
    # UN PAS DU CERVEAU
    # ========================================================

    def step(
        self,
        forced_spikes=None,
    ):

        # ----------------------------------------------------
        # 1. Livraison des événements retardés
        # ----------------------------------------------------

        slot = (
            self.step_index
            % self.buffer_length
        )


        pending = (
            self.delay_buffer[
                slot
            ]
        )


        self.g += pending

        pending.fill(
            0.0
        )


        # ----------------------------------------------------
        # 2. Refractory
        # ----------------------------------------------------

        refractory = (
            self.refractory_remaining
            > 0.0
        )


        active = ~refractory


        # ----------------------------------------------------
        # 3. Intégration LIF
        #
        # Pendant refractory :
        # v et g sont figés.
        #
        # Les entrées peuvent cependant s'accumuler dans g.
        # ----------------------------------------------------

        if active.any():

            old_v = (
                self.v[
                    active
                ].copy()
            )

            old_g = (
                self.g[
                    active
                ].copy()
            )


            self.v[
                active
            ] = (
                self.v_rest
                +
                (
                    old_v
                    - self.v_rest
                )
                * self.a
                +
                old_g
                * self.c
            )


            self.g[
                active
            ] = (
                old_g
                * self.b
            )


        # ----------------------------------------------------
        # 4. Avancer le refractory
        # ----------------------------------------------------

        self.refractory_remaining[
            refractory
        ] -= self.dt_ms


        np.maximum(
            self.refractory_remaining,
            0.0,
            out=self.refractory_remaining,
        )


        # ----------------------------------------------------
        # 5. Spikes produits naturellement par le réseau
        # ----------------------------------------------------

        intrinsic_spikes = np.flatnonzero(
            active
            &
            (
                self.v
                >
                self.v_threshold
            )
        )


        # ----------------------------------------------------
        # 6. Spikes sensoriels imposés
        #
        # Ils serviront plus tard aux haltères,
        # photorécepteurs, etc.
        # ----------------------------------------------------

        if forced_spikes is None:

            forced = np.empty(
                0,
                dtype=np.int64,
            )

        else:

            forced = np.asarray(
                forced_spikes,
                dtype=np.int64,
            )

            forced = forced[
                (
                    forced >= 0
                )
                &
                (
                    forced
                    < self.num_neurons
                )
            ]

            forced = np.unique(
                forced
            )


        if len(forced):

            all_spikes = np.union1d(
                intrinsic_spikes,
                forced,
            )

        else:

            all_spikes = (
                intrinsic_spikes
            )


        # ----------------------------------------------------
        # 7. RESET APRES SPIKE
        # ----------------------------------------------------

        if len(all_spikes):

            self.v[
                all_spikes
            ] = self.v_reset

            self.g[
                all_spikes
            ] = 0.0


            # Spikes intrinsèques :
            # refractory biologique normal.
            if len(intrinsic_spikes):

                self.refractory_remaining[
                    intrinsic_spikes
                ] = (
                    self.refractory_period_ms
                )


            # Pour les neurones forcés par le générateur
            # sensoriel, on autorise immédiatement les futurs
            # événements Poisson, comme dans le protocole
            # de stimulation de référence.
            if len(forced):

                self.refractory_remaining[
                    forced
                ] = 0.0


        # ----------------------------------------------------
        # 8. PROPAGATION
        # ----------------------------------------------------

        propagated_edges = (
            self._schedule_spikes(
                all_spikes
            )
        )


        self.step_index += 1


        return (
            all_spikes,
            propagated_edges,
        )