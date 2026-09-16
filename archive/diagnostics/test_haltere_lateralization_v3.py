import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


CACHE = "data/male_cns/cache_v3"

DURATION_MS = 300.0
STIMULATION_HZ = 100.0


# ============================================================
# CERVEAU
# ============================================================

brain = MaleCNSLIF(
    cache_dir=CACHE,
    dt_ms=0.5,
    global_gain=0.65,
)


metadata = pd.read_feather(
    f"{CACHE}/neurons.feather"
)


# ============================================================
# DETECTION GAUCHE / DROITE
# ============================================================

def infer_side(row):

    # Priorité aux annotations anatomiques
    for column in [
        "somaSide",
        "rootSide",
    ]:

        if column not in row.index:
            continue

        value = row[column]

        if pd.isna(value):
            continue

        value = str(value).strip().upper()

        if value in ("L", "LEFT"):
            return "L"

        if value in ("R", "RIGHT"):
            return "R"


    # Fallback : nom de l'instance
    instance = row.get(
        "instance",
        ""
    )

    if pd.notna(instance):

        instance = str(
            instance
        ).strip().upper()

        if instance.endswith("_L"):
            return "L"

        if instance.endswith("_R"):
            return "R"


    return "?"


# ============================================================
# HALTERES
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


haltere_sides = np.array(
    [
        infer_side(
            metadata.iloc[index]
        )
        for index in haltere_indices
    ]
)


left_haltere = haltere_indices[
    haltere_sides == "L"
]

right_haltere = haltere_indices[
    haltere_sides == "R"
]

unknown_haltere = haltere_indices[
    haltere_sides == "?"
]


print()
print("=" * 70)
print("HALTERE POPULATIONS")
print("=" * 70)

print(
    "Total   :",
    len(haltere_indices)
)

print(
    "Gauche  :",
    len(left_haltere)
)

print(
    "Droite  :",
    len(right_haltere)
)

print(
    "Inconnu :",
    len(unknown_haltere)
)


# ============================================================
# DN / MOTOR
# ============================================================

dn_indices = np.flatnonzero(

    metadata["superclass"]
    .fillna("")
    .astype(str)
    .eq("descending_neuron")
    .to_numpy()
)


motor_indices = np.flatnonzero(

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


print()
print(
    "Descending :",
    len(dn_indices)
)

print(
    "Motor      :",
    len(motor_indices)
)


# ============================================================
# UNE EXPERIENCE
# ============================================================

def run_condition(
    stimulation_indices,
    seed,
):

    brain.reset()

    rng = np.random.default_rng(
        seed
    )


    spike_counts = np.zeros(
        brain.num_neurons,
        dtype=np.int32,
    )


    probability = (
        STIMULATION_HZ
        * brain.dt_ms
        / 1000.0
    )


    steps = int(
        DURATION_MS
        / brain.dt_ms
    )


    forced_total = 0


    for _ in range(steps):

        random_values = rng.random(
            len(stimulation_indices)
        )


        forced = stimulation_indices[
            random_values
            <
            probability
        ]


        forced_total += len(
            forced
        )


        spikes, _ = brain.step(
            forced_spikes=forced
        )


        if len(spikes):

            spike_counts[
                spikes
            ] += 1


    return (
        spike_counts,
        forced_total,
    )


# ============================================================
# GAUCHE
# ============================================================

print()
print("Simulation HALTERE GAUCHE...")

left_counts, left_forced = (
    run_condition(
        left_haltere,
        seed=42,
    )
)


# ============================================================
# DROITE
# ============================================================

print(
    "Simulation HALTERE DROIT..."
)

right_counts, right_forced = (
    run_condition(
        right_haltere,
        seed=42,
    )
)


# ============================================================
# NORMALISATION
#
# spikes downstream pour 1000 spikes sensoriels
# ============================================================

left_normalized = (
    left_counts.astype(
        np.float64
    )
    *
    1000.0
    /
    max(
        left_forced,
        1,
    )
)


right_normalized = (
    right_counts.astype(
        np.float64
    )
    *
    1000.0
    /
    max(
        right_forced,
        1,
    )
)


# ============================================================
# RESUME
# ============================================================

print()
print("=" * 70)
print("RESULTATS")
print("=" * 70)


print(
    "Spikes forces gauche :",
    left_forced
)

print(
    "Spikes forces droite :",
    right_forced
)


print()
print(
    "DN spikes gauche :",
    int(
        left_counts[
            dn_indices
        ].sum()
    )
)

print(
    "DN spikes droite :",
    int(
        right_counts[
            dn_indices
        ].sum()
    )
)


print()
print(
    "Motor spikes gauche :",
    int(
        left_counts[
            motor_indices
        ].sum()
    )
)

print(
    "Motor spikes droite :",
    int(
        right_counts[
            motor_indices
        ].sum()
    )
)


print()
print(
    "DN / 1000 input gauche :",
    float(
        left_normalized[
            dn_indices
        ].sum()
    )
)

print(
    "DN / 1000 input droite :",
    float(
        right_normalized[
            dn_indices
        ].sum()
    )
)


print()
print(
    "Motor / 1000 input gauche :",
    float(
        left_normalized[
            motor_indices
        ].sum()
    )
)

print(
    "Motor / 1000 input droite :",
    float(
        right_normalized[
            motor_indices
        ].sum()
    )
)


# ============================================================
# DN LES PLUS ASYMETRIQUES
# ============================================================

dn_difference = (
    left_normalized[
        dn_indices
    ]
    -
    right_normalized[
        dn_indices
    ]
)


order = np.argsort(
    np.abs(
        dn_difference
    )
)[::-1]


print()
print("=" * 70)
print("DESCENDING NEURONS LES PLUS ASYMETRIQUES")
print("=" * 70)


shown = 0


for local_position in order:

    neuron_index = (
        dn_indices[
            local_position
        ]
    )

    L = float(
        left_normalized[
            neuron_index
        ]
    )

    R = float(
        right_normalized[
            neuron_index
        ]
    )

    delta = L - R


    if (
        L == 0.0
        and
        R == 0.0
    ):
        continue


    row = metadata.iloc[
        neuron_index
    ]


    print(
        f"{shown + 1:02d} | "
        f"body={row['bodyId']} | "
        f"type={row['type']} | "
        f"instance={row['instance']} | "
        f"L={L:.2f} | "
        f"R={R:.2f} | "
        f"delta={delta:+.2f}"
    )


    shown += 1


    if shown >= 30:
        break


# ============================================================
# MOTOR NEURONS LES PLUS ASYMETRIQUES
# ============================================================

motor_difference = (
    left_normalized[
        motor_indices
    ]
    -
    right_normalized[
        motor_indices
    ]
)


order = np.argsort(
    np.abs(
        motor_difference
    )
)[::-1]


print()
print("=" * 70)
print("MOTOR NEURONS LES PLUS ASYMETRIQUES")
print("=" * 70)


shown = 0


for local_position in order:

    neuron_index = (
        motor_indices[
            local_position
        ]
    )

    L = float(
        left_normalized[
            neuron_index
        ]
    )

    R = float(
        right_normalized[
            neuron_index
        ]
    )

    delta = L - R


    if (
        L == 0.0
        and
        R == 0.0
    ):
        continue


    row = metadata.iloc[
        neuron_index
    ]


    print(
        f"{shown + 1:02d} | "
        f"body={row['bodyId']} | "
        f"type={row['type']} | "
        f"instance={row['instance']} | "
        f"L={L:.2f} | "
        f"R={R:.2f} | "
        f"delta={delta:+.2f}"
    )


    shown += 1


    if shown >= 30:
        break