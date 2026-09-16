import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


CACHE = "data/male_cns/cache_v3"

DURATION_MS = 300.0
N_TRIALS = 5

BASE_RATE = 75.0

DIFFERENTIALS = [
    -50.0,
    -25.0,
    0.0,
    25.0,
    50.0,
]


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

        value = row.get(
            column,
            None,
        )

        if pd.isna(value):
            continue

        value = str(value).upper()

        if value in ("L", "LEFT"):
            return "L"

        if value in ("R", "RIGHT"):
            return "R"


    instance = row.get(
        "instance",
        "",
    )

    if pd.notna(instance):

        instance = str(
            instance
        ).upper()

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


# ============================================================
# b1 / b2
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


up_mask = (
    type_clean.isin(
        [
            "b1mn",
            "b2mn",
        ]
    )
    .to_numpy()
)

up_indices = np.flatnonzero(
    up_mask
)

up_sides = np.array(
    [
        infer_side(
            metadata.iloc[index]
        )
        for index in up_indices
    ]
)

up_left = up_indices[
    up_sides == "L"
]

up_right = up_indices[
    up_sides == "R"
]


print()
print("Haltere L :", len(left_haltere))
print("Haltere R :", len(right_haltere))

print(
    "b1/b2 L :",
    len(up_left)
)

print(
    "b1/b2 R :",
    len(up_right)
)


# ============================================================
# RUN
# ============================================================

def run_trial(
    left_rate,
    right_rate,
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

    steps = int(
        DURATION_MS
        / brain.dt_ms
    )


    p_left = (
        left_rate
        * brain.dt_ms
        / 1000.0
    )

    p_right = (
        right_rate
        * brain.dt_ms
        / 1000.0
    )


    for _ in range(steps):

        forced_left = left_haltere[
            rng.random(
                len(left_haltere)
            )
            < p_left
        ]

        forced_right = right_haltere[
            rng.random(
                len(right_haltere)
            )
            < p_right
        ]


        forced = np.concatenate(
            [
                forced_left,
                forced_right,
            ]
        )


        spikes, _ = brain.step(
            forced_spikes=forced
        )


        if len(spikes):

            counts[
                spikes
            ] += 1


    left_output = float(
        counts[
            up_left
        ].sum()
    )

    right_output = float(
        counts[
            up_right
        ].sum()
    )


    # Signal de roulis fourni par le cerveau
    roll_signal = (
        left_output
        -
        right_output
    )


    return (
        left_output,
        right_output,
        roll_signal,
    )


# ============================================================
# SWEEP
# ============================================================

print()
print("=" * 70)
print("ROLL RESPONSE")
print("=" * 70)


for differential in DIFFERENTIALS:

    left_rate = (
        BASE_RATE
        + differential
    )

    right_rate = (
        BASE_RATE
        - differential
    )


    trials = []


    for trial in range(
        N_TRIALS
    ):

        result = run_trial(

            left_rate=left_rate,

            right_rate=right_rate,

            seed=(
                1000
                + trial
            ),
        )

        trials.append(
            result
        )


    trials = np.asarray(
        trials,
        dtype=np.float64,
    )


    L = trials[:, 0]
    R = trials[:, 1]
    roll = trials[:, 2]


    print()

    print(
        f"delta={differential:+5.0f} Hz | "
        f"L_rate={left_rate:5.0f} | "
        f"R_rate={right_rate:5.0f}"
    )

    print(
        f"    b1/b2 L : "
        f"{L.mean():6.2f}"
        f" +/- {L.std():5.2f}"
    )

    print(
        f"    b1/b2 R : "
        f"{R.mean():6.2f}"
        f" +/- {R.std():5.2f}"
    )

    print(
        f"    ROLL    : "
        f"{roll.mean():+7.2f}"
        f" +/- {roll.std():5.2f}"
    )