import time

import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


CACHE = "data/male_cns/cache_v3"


# ============================================================
# CHARGEMENT
# ============================================================

brain = MaleCNSLIF(
    cache_dir=CACHE,

    # Rapide pour notre premier test.
    dt_ms=0.5,

    global_gain=0.65,
)


metadata = pd.read_feather(
    f"{CACHE}/neurons.feather"
)


# ============================================================
# POPULATIONS
# ============================================================

haltere_mask = (
    metadata["subclass"]
    .fillna("")
    .astype(str)
    .str.lower()
    .eq("haltere")
    .to_numpy()
)


haltere_indices = np.flatnonzero(
    haltere_mask
)


dn_mask = (
    metadata["superclass"]
    .fillna("")
    .astype(str)
    .eq("descending_neuron")
    .to_numpy()
)


dn_indices = np.flatnonzero(
    dn_mask
)


motor_mask = (
    metadata["superclass"]
    .fillna("")
    .astype(str)
    .isin(
        [
            "vnc_motor",
            "cb_motor",
        ]
    )
    .to_numpy()
)


motor_indices = np.flatnonzero(
    motor_mask
)


print()
print("=" * 60)
print("POPULATIONS")
print("=" * 60)

print(
    "Haltere neurons :",
    len(haltere_indices)
)

print(
    "Descending      :",
    len(dn_indices)
)

print(
    "Motor neurons   :",
    len(motor_indices)
)


# ============================================================
# TEST 1
#
# Un cerveau au repos doit rester silencieux.
# ============================================================

print()
print("=" * 60)
print("TEST 1 - CERVEAU SANS STIMULATION")
print("=" * 60)


brain.reset()


baseline_ms = 50.0

baseline_steps = int(
    baseline_ms
    / brain.dt_ms
)


baseline_spikes = 0


start = time.perf_counter()


for _ in range(
    baseline_steps
):

    spikes, edges = brain.step()

    baseline_spikes += len(
        spikes
    )


elapsed = (
    time.perf_counter()
    - start
)


print(
    "Temps simule :",
    baseline_ms,
    "ms"
)

print(
    "Spikes :",
    baseline_spikes
)

print(
    "Temps reel :",
    f"{elapsed:.3f} s"
)


# ============================================================
# TEST 2
#
# Stimulation Poisson des haltères.
# ============================================================

print()
print("=" * 60)
print("TEST 2 - STIMULATION HALTERE")
print("=" * 60)


brain.reset()


rng = np.random.default_rng(
    42
)


STIMULATION_HZ = 100.0

DURATION_MS = 200.0


steps = int(
    DURATION_MS
    / brain.dt_ms
)


# Probabilité d'un spike sensoriel
# pendant un timestep.

spike_probability = (
    STIMULATION_HZ
    * brain.dt_ms
    / 1000.0
)


spike_counts = np.zeros(
    brain.num_neurons,
    dtype=np.int32,
)


forced_total = 0

propagated_edges_total = 0


start = time.perf_counter()


for step in range(
    steps
):

    random_values = rng.random(
        len(
            haltere_indices
        )
    )


    forced = haltere_indices[
        random_values
        <
        spike_probability
    ]


    forced_total += len(
        forced
    )


    (
        spikes,
        propagated_edges,
    ) = brain.step(
        forced_spikes=forced
    )


    propagated_edges_total += (
        propagated_edges
    )


    if len(spikes):

        spike_counts[
            spikes
        ] += 1


elapsed = (
    time.perf_counter()
    - start
)


# ============================================================
# RESULTATS
# ============================================================

total_spikes = int(
    spike_counts.sum()
)


active_neurons = int(
    np.count_nonzero(
        spike_counts
    )
)


dn_spikes = int(
    spike_counts[
        dn_indices
    ].sum()
)


motor_spikes = int(
    spike_counts[
        motor_indices
    ].sum()
)


print(
    "Temps simule :",
    DURATION_MS,
    "ms"
)

print(
    "Temps reel   :",
    f"{elapsed:.3f} s"
)

print(
    "Real-time factor :",
    (
        DURATION_MS
        / 1000.0
    )
    /
    max(
        elapsed,
        1e-9,
    )
)


print()
print(
    "Spikes forces haltere :",
    forced_total
)

print(
    "Spikes total cerveau  :",
    total_spikes
)

print(
    "Neurones actifs       :",
    active_neurons,
    "/",
    brain.num_neurons
)

print(
    "Spikes DN             :",
    dn_spikes
)

print(
    "Spikes moteurs        :",
    motor_spikes
)

print(
    "Aretes propagees      :",
    propagated_edges_total
)


# ============================================================
# TOP DESCENDING NEURONS
# ============================================================

dn_counts = (
    spike_counts[
        dn_indices
    ]
)


order = np.argsort(
    dn_counts
)[::-1]


print()
print("=" * 60)
print("TOP DESCENDING NEURONS")
print("=" * 60)


shown = 0


for local_position in order:

    count = int(
        dn_counts[
            local_position
        ]
    )


    if count <= 0:
        break


    neuron_index = (
        dn_indices[
            local_position
        ]
    )


    row = metadata.iloc[
        neuron_index
    ]


    firing_rate = (
        count
        /
        (
            DURATION_MS
            / 1000.0
        )
    )


    print(
        f"{shown + 1:02d} | "
        f"body={row['bodyId']} | "
        f"type={row['type']} | "
        f"instance={row['instance']} | "
        f"spikes={count} | "
        f"rate={firing_rate:.1f} Hz"
    )


    shown += 1


    if shown >= 20:
        break