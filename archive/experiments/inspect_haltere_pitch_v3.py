from pathlib import Path

import numpy as np
import pandas as pd


CACHE = Path(
    "data/male_cns/cache_v3"
)

OUTPUT_DIR = Path(
    "logs/pitch"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# LOAD CACHE
# ============================================================

print(
    "Chargement du cache MaleCNS..."
)

metadata = pd.read_feather(
    CACHE / "neurons.feather"
)

body_ids = np.load(
    CACHE / "body_ids.npy",
    mmap_mode="r",
)

row_ptr = np.load(
    CACHE / "row_ptr.npy",
    mmap_mode="r",
)

post_idx = np.load(
    CACHE / "post_idx.npy",
    mmap_mode="r",
)

syn_count = np.load(
    CACHE / "syn_count.npy",
    mmap_mode="r",
)


# ============================================================
# ALIGN METADATA -> CACHE INDEX
# ============================================================

body_to_index = {
    int(body_id): index
    for index, body_id
    in enumerate(body_ids)
}


metadata[
    "_cache_index"
] = (
    metadata["bodyId"]
    .map(body_to_index)
)


if metadata[
    "_cache_index"
].isna().any():

    raise RuntimeError(
        "Certains bodyId metadata "
        "sont absents du cache."
    )


metadata[
    "_cache_index"
] = (
    metadata[
        "_cache_index"
    ]
    .astype(np.int64)
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

        value = (
            str(value)
            .strip()
            .upper()
        )

        if value in (
            "L",
            "LEFT",
        ):
            return "L"

        if value in (
            "R",
            "RIGHT",
        ):
            return "R"


    instance = row.get(
        "instance",
        "",
    )

    if pd.notna(instance):

        instance = (
            str(instance)
            .strip()
            .upper()
        )

        if instance.endswith(
            "_L"
        ):
            return "L"

        if instance.endswith(
            "_R"
        ):
            return "R"


    return "?"


metadata[
    "_side"
] = metadata.apply(
    infer_side,
    axis=1,
)


# ============================================================
# HALTERE POPULATION
# ============================================================

haltere_mask = (
    metadata["subclass"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("haltere")
)


haltere = (
    metadata[
        haltere_mask
    ]
    .copy()
)


haltere[
    "_type"
] = (
    haltere["type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


haltere.loc[
    haltere["_type"] == "",
    "_type",
] = "UNLABELED"


print()
print(
    "=" * 76
)

print(
    "HALTERE TYPES"
)

print(
    "=" * 76
)

print(
    "Total :",
    len(haltere)
)


# ============================================================
# TYPE COUNTS
# ============================================================

type_rows = []


for type_name, group in (
    haltere.groupby(
        "_type"
    )
):

    left = int(
        (
            group["_side"]
            == "L"
        ).sum()
    )

    right = int(
        (
            group["_side"]
            == "R"
        ).sum()
    )

    unknown = int(
        (
            group["_side"]
            == "?"
        ).sum()
    )


    instances = (
        group["instance"]
        .dropna()
        .astype(str)
        .head(4)
        .tolist()
    )


    type_rows.append(
        {
            "type":
                type_name,

            "total":
                len(group),

            "left":
                left,

            "right":
                right,

            "unknown":
                unknown,

            "examples":
                " | ".join(
                    instances
                ),
        }
    )


type_table = pd.DataFrame(
    type_rows
).sort_values(
    [
        "total",
        "type",
    ],
    ascending=[
        False,
        True,
    ],
)


print()

print(
    type_table.to_string(
        index=False
    )
)


type_file = (
    OUTPUT_DIR
    /
    "haltere_types.csv"
)


type_table.to_csv(
    type_file,
    index=False,
)


# ============================================================
# FLIGHT MOTOR NEURONS
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


FLIGHT_MOTORS = {
    "b1": "b1mn",
    "b2": "b2mn",
    "b3": "b3mn",
    "i1": "i1mn",
    "i2": "i2mn",
}


motor_indices = {}


for motor_name, target_type in (
    FLIGHT_MOTORS.items()
):

    indices = np.flatnonzero(
        (
            type_clean
            == target_type
        ).to_numpy()
    )


    for metadata_index in indices:

        row = metadata.iloc[
            metadata_index
        ]

        side = infer_side(
            row
        )


        if side not in (
            "L",
            "R",
        ):
            continue


        cache_index = int(
            row[
                "_cache_index"
            ]
        )


        key = (
            motor_name
            + "_"
            + side
        )


        motor_indices[
            key
        ] = (
            cache_index
        )


print()
print(
    "=" * 76
)

print(
    "FLIGHT MOTOR TARGETS"
)

print(
    "=" * 76
)


for key in sorted(
    motor_indices
):

    cache_index = (
        motor_indices[
            key
        ]
    )

    body_id = int(
        body_ids[
            cache_index
        ]
    )

    print(
        f"{key:5s}"
        f" -> body={body_id}"
        f" cache_index={cache_index}"
    )


# ============================================================
# DIRECT CONNECTIVITY
#
# HALTERE TYPE/SIDE -> b1,b2,b3,i1,i2
# ============================================================

print()
print(
    "=" * 76
)

print(
    "DIRECT HALTERE -> FLIGHT MOTOR CONNECTIVITY"
)

print(
    "=" * 76
)


rows = []


for (
    haltere_type,
    haltere_side,
), group in haltere.groupby(
    [
        "_type",
        "_side",
    ]
):

    if haltere_side not in (
        "L",
        "R",
    ):
        continue


    pre_indices = (
        group[
            "_cache_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    result = {
        "haltere_type":
            haltere_type,

        "haltere_side":
            haltere_side,

        "n_neurons":
            len(pre_indices),
    }


    # Initialise sorties
    for target_name in sorted(
        motor_indices
    ):

        result[
            target_name
        ] = 0


    # ------------------------------------------
    # Pour chaque neurone sensoriel du groupe
    # ------------------------------------------

    for pre in pre_indices:

        start = int(
            row_ptr[
                pre
            ]
        )

        end = int(
            row_ptr[
                pre + 1
            ]
        )


        targets = post_idx[
            start:end
        ]

        weights = syn_count[
            start:end
        ]


        for (
            target_name,
            target_index,
        ) in motor_indices.items():

            mask = (
                targets
                == target_index
            )


            if np.any(
                mask
            ):

                result[
                    target_name
                ] += int(
                    weights[
                        mask
                    ].sum()
                )


    rows.append(
        result
    )


connection_table = (
    pd.DataFrame(
        rows
    )
)


motor_columns = sorted(
    motor_indices.keys()
)


connection_table[
    "total_direct"
] = (
    connection_table[
        motor_columns
    ]
    .sum(
        axis=1
    )
)


connection_table[
    "direct_per_neuron"
] = (
    connection_table[
        "total_direct"
    ]
    /
    connection_table[
        "n_neurons"
    ]
)


connection_table = (
    connection_table
    .sort_values(
        "total_direct",
        ascending=False,
    )
)


print()

print(
    connection_table.to_string(
        index=False
    )
)


connection_file = (
    OUTPUT_DIR
    /
    "haltere_to_flight_motors.csv"
)


connection_table.to_csv(
    connection_file,
    index=False,
)


# ============================================================
# MOST INTERESTING TYPES
# ============================================================

print()
print(
    "=" * 76
)

print(
    "TOP HALTERE CHANNELS"
)

print(
    "=" * 76
)


for _, row in (
    connection_table
    .head(20)
    .iterrows()
):

    print()

    print(
        f"{row['haltere_type']} "
        f"{row['haltere_side']} "
        f"| n={row['n_neurons']} "
        f"| direct={row['total_direct']}"
    )


    values = []


    for motor in (
        motor_columns
    ):

        weight = int(
            row[
                motor
            ]
        )


        if weight > 0:

            values.append(
                (
                    motor,
                    weight,
                )
            )


    values.sort(
        key=lambda x: x[1],
        reverse=True,
    )


    if values:

        print(
            "   "
            +
            " | ".join(
                f"{motor}={weight}"
                for motor, weight
                in values
            )
        )

    else:

        print(
            "   aucun lien direct "
            ">= seuil cache"
        )


# ============================================================
# FILES
# ============================================================

print()
print(
    "=" * 76
)

print(
    "FICHIERS"
)

print(
    "=" * 76
)

print(
    type_file
)

print(
    connection_file
)