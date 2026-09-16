from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather


# ============================================================
# PATHS
# ============================================================

DATA = Path("data/male_cns")

ANNOTATIONS = DATA / (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

NEUROTRANSMITTERS = DATA / (
    "body-neurotransmitters-male-cns-v1.0.feather"
)

CONNECTOME = DATA / (
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)

OUT = DATA / "cache_v3"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIG
# ============================================================

MIN_WEIGHT = 5


# ============================================================
# ANNOTATIONS
# ============================================================

print("Chargement annotations...")

ann = feather.read_feather(
    ANNOTATIONS
)


# On ne garde QUE les vrais neurones traces
traced = ann[
    ann["status"]
    .fillna("")
    .astype(str)
    .eq("Traced")
].copy()


traced = (
    traced
    .drop_duplicates(
        subset=["bodyId"]
    )
    .reset_index(drop=True)
)


body_ids = (
    traced["bodyId"]
    .astype(np.int64)
    .sort_values()
    .to_numpy()
)


NUM_NEURONS = len(
    body_ids
)


print(
    "Neurones Traced :",
    NUM_NEURONS
)


# ============================================================
# FONCTION DE LOOKUP RAPIDE
#
# bodyId -> index dense
# ============================================================

def lookup_indices(values):

    values = np.asarray(
        values,
        dtype=np.int64,
    )

    positions = np.searchsorted(
        body_ids,
        values,
    )

    valid = (
        positions
        < NUM_NEURONS
    )


    result_valid = np.zeros(
        len(values),
        dtype=bool,
    )


    if valid.any():

        valid_positions = positions[
            valid
        ]

        result_valid[
            valid
        ] = (
            body_ids[
                valid_positions
            ]
            ==
            values[
                valid
            ]
        )


    return positions, result_valid


# ============================================================
# SCAN DU CONNECTOME
# ============================================================

print()
print("Scan connectome MaleCNS...")

source = pa.memory_map(
    str(CONNECTOME),
    "r",
)

reader = ipc.open_file(
    source
)


all_pre_idx = []
all_post_idx = []
all_weights = []


for i in range(
    reader.num_record_batches
):

    batch = reader.get_batch(i)


    pre = (
        batch.column(
            batch.schema.get_field_index(
                "body_pre"
            )
        )
        .to_numpy(
            zero_copy_only=False
        )
        .astype(
            np.int64,
            copy=False,
        )
    )


    post = (
        batch.column(
            batch.schema.get_field_index(
                "body_post"
            )
        )
        .to_numpy(
            zero_copy_only=False
        )
        .astype(
            np.int64,
            copy=False,
        )
    )


    weight = (
        batch.column(
            batch.schema.get_field_index(
                "weight"
            )
        )
        .to_numpy(
            zero_copy_only=False
        )
    )


    # --------------------------------------------------------
    # Lookup dans la population Traced
    # --------------------------------------------------------

    pre_idx, pre_valid = (
        lookup_indices(pre)
    )

    post_idx, post_valid = (
        lookup_indices(post)
    )


    keep = (
        pre_valid
        &
        post_valid
        &
        (weight >= MIN_WEIGHT)
    )


    if keep.any():

        all_pre_idx.append(
            pre_idx[
                keep
            ].astype(
                np.int32
            )
        )

        all_post_idx.append(
            post_idx[
                keep
            ].astype(
                np.int32
            )
        )

        all_weights.append(
            weight[
                keep
            ].astype(
                np.int32
            )
        )


    if (
        i % 200 == 0
        or
        i == reader.num_record_batches - 1
    ):

        print(
            f"  batch "
            f"{i + 1}/"
            f"{reader.num_record_batches}"
        )


# ============================================================
# ASSEMBLAGE
# ============================================================

print()
print(
    "Assemblage..."
)


pre_idx = np.concatenate(
    all_pre_idx
)

post_idx = np.concatenate(
    all_post_idx
)

syn_count = np.concatenate(
    all_weights
)


NUM_CONNECTIONS = len(
    pre_idx
)


print(
    "Connexions conservees :",
    NUM_CONNECTIONS
)


# ============================================================
# TRI CSR
#
# Les connexions sortantes d'un neurone seront contiguës.
# ============================================================

print()
print(
    "Construction CSR..."
)


order = np.argsort(
    pre_idx,
    kind="stable",
)


pre_idx = pre_idx[
    order
]

post_idx = post_idx[
    order
]

syn_count = syn_count[
    order
]


counts = np.bincount(
    pre_idx,
    minlength=NUM_NEURONS,
)


row_ptr = np.zeros(
    NUM_NEURONS + 1,
    dtype=np.int64,
)


row_ptr[1:] = np.cumsum(
    counts,
    dtype=np.int64,
)


# ============================================================
# METADATA
# ============================================================

wanted_columns = [
    "bodyId",
    "type",
    "instance",

    "status",
    "statusLabel",

    "superclass",
    "class",
    "subclass",

    "somaSide",
    "rootSide",
    "somaNeuromere",

    "entryNerve",
    "exitNerve",

    "assignedOlHex1",
    "assignedOlHex2",
]


wanted_columns = [
    column
    for column in wanted_columns
    if column in traced.columns
]


metadata = pd.DataFrame(
    {
        "bodyId": body_ids
    }
)


metadata = metadata.merge(
    traced[
        wanted_columns
    ],
    on="bodyId",
    how="left",
)


# ============================================================
# NEUROTRANSMETTEURS
# ============================================================

print()
print(
    "Chargement neurotransmetteurs..."
)


nt = feather.read_feather(
    NEUROTRANSMITTERS
)


# Plusieurs lignes peuvent exister pour le même body.
# On prend ici la prédiction individuelle la plus confiante.

nt = nt.sort_values(
    "predicted_nt_confidence",
    ascending=False,
)


nt_best = (
    nt
    .drop_duplicates(
        subset=["body"]
    )
    .copy()
)


def clean_nt(value):

    if pd.isna(value):
        return "unclear"

    value = (
        str(value)
        .strip()
        .lower()
    )

    if value in (
        "",
        "nan",
        "none",
    ):
        return "unclear"

    return value


resolved = []


for row in nt_best.itertuples():

    consensus = clean_nt(
        getattr(
            row,
            "consensus_nt",
            None,
        )
    )

    predicted = clean_nt(
        getattr(
            row,
            "predicted_nt",
            None,
        )
    )


    if consensus != "unclear":

        resolved.append(
            consensus
        )

    elif predicted != "unclear":

        resolved.append(
            predicted
        )

    else:

        resolved.append(
            "unclear"
        )


nt_best[
    "resolved_nt"
] = resolved


metadata = metadata.merge(

    nt_best[
        [
            "body",
            "resolved_nt",
            "predicted_nt_confidence",
        ]
    ],

    left_on="bodyId",
    right_on="body",

    how="left",
)


metadata.drop(
    columns=["body"],
    inplace=True,
    errors="ignore",
)


metadata[
    "resolved_nt"
] = (
    metadata[
        "resolved_nt"
    ]
    .fillna(
        "unclear"
    )
    .astype(str)
    .str.lower()
)


# ============================================================
# NT CODE
#
# Pour le moment on ne décide PAS encore du comportement
# électrique exact.
#
# On encode seulement l'identité du neurotransmetteur.
# ============================================================

NT_CODES = {

    "unclear": 0,

    "acetylcholine": 1,

    "gaba": 2,

    "glutamate": 3,

    "histamine": 4,

    "serotonin": 5,

    "dopamine": 6,

    "octopamine": 7,

    "tyramine": 8,
}


nt_code = (
    metadata[
        "resolved_nt"
    ]
    .map(
        NT_CODES
    )
    .fillna(0)
    .astype(
        np.int8
    )
    .to_numpy()
)


# ============================================================
# SAUVEGARDE
# ============================================================

print()
print(
    "Sauvegarde cache V3..."
)


np.save(
    OUT / "body_ids.npy",
    body_ids,
)


np.save(
    OUT / "row_ptr.npy",
    row_ptr,
)


np.save(
    OUT / "post_idx.npy",
    post_idx,
)


np.save(
    OUT / "syn_count.npy",
    syn_count,
)


np.save(
    OUT / "nt_code.npy",
    nt_code,
)


metadata.reset_index(
    drop=True
).to_feather(
    OUT
    / "neurons.feather"
)


manifest = {

    "version":
        "neuroflight-malecns-v3",

    "population":
        "status=Traced",

    "min_weight":
        MIN_WEIGHT,

    "num_neurons":
        int(
            NUM_NEURONS
        ),

    "num_connections":
        int(
            NUM_CONNECTIONS
        ),

    "source_connectome":
        CONNECTOME.name,

    "source_annotations":
        ANNOTATIONS.name,

    "source_neurotransmitters":
        NEUROTRANSMITTERS.name,

    "nt_codes":
        NT_CODES,
}


with open(
    OUT / "manifest.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        manifest,
        file,
        indent=2,
    )


# ============================================================
# RESUME
# ============================================================

print()
print(
    "=" * 65
)

print(
    " NEUROFLIGHT MALECNS V3"
)

print(
    "=" * 65
)


print(
    "Neurones Traced :",
    NUM_NEURONS
)


print(
    "Connexions      :",
    NUM_CONNECTIONS
)


print(
    "Seuil synapses  :",
    MIN_WEIGHT
)


print()
print(
    "Neurotransmetteurs :"
)


print(
    metadata[
        "resolved_nt"
    ]
    .value_counts()
)


print()
print(
    "Superclasses principales :"
)


print(
    metadata[
        "superclass"
    ]
    .fillna("<EMPTY>")
    .value_counts()
    .head(20)
)


print()
print(
    "Cache cree dans :",
    OUT
)