import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


CACHE = "data/male_cns/cache_v3"

DURATION_MS = 300.0
STIMULATION_HZ = 100.0

# Plusieurs essais pour éviter de conclure
# à partir d'un hasard du générateur Poisson.
N_TRIALS = 5


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
# SIDE
# ============================================================

def infer_side(row):

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
# HALTERE
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


haltere_sides = np.asarray(
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


# ============================================================
# FLIGHT STEERING MOTOR NEURONS
#
# b1 / b2
# b3 / i1 / i2
# ============================================================

type_clean = (
    metadata["type"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.replace(
        " ",
        "",
        regex=False,
    )
)


motor_name_map = {
    "b1mn": "b1",
    "b2mn": "b2",
    "b3mn": "b3",
    "i1mn": "i1",
    "i2mn": "i2",
}


flight_motor_mask = (
    type_clean.isin(
        motor_name_map.keys()
    )
    .to_numpy()
)


flight_motor_indices = np.flatnonzero(
    flight_motor_mask
)


flight_motor_names = np.asarray(
    [
        motor_name_map[
            type_clean.iloc[index]
        ]
        for index in flight_motor_indices
    ]
)


flight_motor_sides = np.asarray(
    [
        infer_side(
            metadata.iloc[index]
        )
        for index in flight_motor_indices
    ]
)


print()
print("=" * 70)
print("FLIGHT MOTOR POPULATION")
print("=" * 70)

print(
    "Haltere gauche :",
    len(left_haltere)
)

print(
    "Haltere droite :",
    len(right_haltere)
)

print()

for name in [
    "b1",
    "b2",
    "b3",
    "i1",
    "i2",
]:

    mask = (
        flight_motor_names
        == name
    )

    print(
        f"{name:>2} :",
        int(mask.sum())
    )


print()
print(
    "Total steering MN :",
    len(flight_motor_indices)
)


# ============================================================
# UNE SIMULATION
# ============================================================

def run_condition(
    stimulation_indices,
    seed,
):

    brain.reset()

    rng = np.random.default_rng(
        seed
    )


    counts = np.zeros(
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

        forced = stimulation_indices[
            rng.random(
                len(stimulation_indices)
            )
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

            counts[
                spikes
            ] += 1


    # Normalisation :
    # activité pour 1000 spikes sensoriels.

    normalized = (
        counts.astype(np.float64)
        *
        1000.0
        /
        max(
            forced_total,
            1,
        )
    )


    return normalized


# ============================================================
# PLUSIEURS ESSAIS
# ============================================================

left_trials = []
right_trials = []


for trial in range(
    N_TRIALS
):

    seed = (
        1000
        + trial
    )


    print(
        f"Essai {trial + 1}/{N_TRIALS}"
    )


    left = run_condition(
        left_haltere,
        seed=seed,
    )


    right = run_condition(
        right_haltere,
        seed=seed,
    )


    left_trials.append(
        left[
            flight_motor_indices
        ]
    )

    right_trials.append(
        right[
            flight_motor_indices
        ]
    )


left_trials = np.asarray(
    left_trials
)

right_trials = np.asarray(
    right_trials
)


# ============================================================
# PAR NEURONE
# ============================================================

left_mean = left_trials.mean(
    axis=0
)

left_std = left_trials.std(
    axis=0
)


right_mean = right_trials.mean(
    axis=0
)

right_std = right_trials.std(
    axis=0
)


difference = (
    left_mean
    - right_mean
)


print()
print("=" * 70)
print("FLIGHT MOTOR NEURONS")
print("=" * 70)


order = np.argsort(
    np.abs(
        difference
    )
)[::-1]


for position in order:

    index = (
        flight_motor_indices[
            position
        ]
    )

    row = metadata.iloc[
        index
    ]


    print(
        f"{flight_motor_names[position]:>2} "
        f"{flight_motor_sides[position]} | "
        f"body={row['bodyId']} | "
        f"L={left_mean[position]:6.2f}"
        f" +/- {left_std[position]:5.2f} | "
        f"R={right_mean[position]:6.2f}"
        f" +/- {right_std[position]:5.2f} | "
        f"delta={difference[position]:+7.2f}"
    )


# ============================================================
# MODULES FONCTIONNELS
#
# On ne les convertit PAS encore en torque.
#
# On mesure seulement leur activité.
# ============================================================

MODULES = {

    "AMPLITUDE_UP": {
        "b1",
        "b2",
    },

    "AMPLITUDE_DOWN": {
        "b3",
        "i1",
        "i2",
    },
}


print()
print("=" * 70)
print("FLIGHT MODULES")
print("=" * 70)


for module_name, names in (
    MODULES.items()
):

    for side in [
        "L",
        "R",
    ]:

        mask = np.asarray(
            [
                (
                    name in names
                    and neuron_side == side
                )
                for (
                    name,
                    neuron_side,
                )
                in zip(
                    flight_motor_names,
                    flight_motor_sides,
                )
            ]
        )


        if not mask.any():
            continue


        L = (
            left_trials[
                :,
                mask
            ]
            .sum(axis=1)
        )

        R = (
            right_trials[
                :,
                mask
            ]
            .sum(axis=1)
        )


        print()

        print(
            module_name,
            "SIDE",
            side,
        )

        print(
            "  stimulation gauche : "
            f"{L.mean():.2f}"
            f" +/- {L.std():.2f}"
        )

        print(
            "  stimulation droite : "
            f"{R.mean():.2f}"
            f" +/- {R.std():.2f}"
        )

        print(
            "  delta : "
            f"{(L.mean() - R.mean()):+.2f}"
        )