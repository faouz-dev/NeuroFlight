from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather


# ============================================================
# FICHIERS
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


# Seuil provisoire :
# on ignore les connexions extrêmement faibles.
#
# Ce n'est PAS une vérité biologique.
# C'est uniquement pour construire un premier réseau tractable.
MIN_WEIGHT = 10


print("Chargement annotations...")

ann = feather.read_feather(ANNOTATIONS)

print("Annotations :", len(ann))


# ============================================================
# 1. ENTREES SENSORIELLES
# ============================================================

types = (
    ann["type"]
    .fillna("")
    .astype(str)
)

superclass = (
    ann["superclass"]
    .fillna("")
    .astype(str)
)

subclass = (
    ann["subclass"]
    .fillna("")
    .astype(str)
)


# ------------------------------------------------------------
# Wide-field visual neurons
#
# VS et HS sont particulièrement intéressants
# pour les informations de mouvement visuel.
# ------------------------------------------------------------

visual_mask = types.isin(
    [
        "VS",
        "HSN",
        "HSE",
        "HSS",
    ]
)


# ------------------------------------------------------------
# Halteres
# ------------------------------------------------------------

haltere_mask = (
    subclass
    .str.lower()
    .eq("haltere")
)


sensory = ann[
    visual_mask | haltere_mask
].copy()


visual = ann[
    visual_mask
].copy()

haltere = ann[
    haltere_mask
].copy()


print()
print("Neurones visuels VS/HS :", len(visual))
print("Neurones haltere       :", len(haltere))
print("Seeds sensoriels total :", len(sensory))


# ============================================================
# 2. DESCENDING NEURONS
# ============================================================

descending = ann[
    superclass == "descending_neuron"
].copy()

print(
    "Descending neurons    :",
    len(descending)
)


# ============================================================
# 3. MOTOR NEURONS
# ============================================================

motor = ann[
    superclass.isin(
        [
            "vnc_motor",
            "cb_motor",
        ]
    )
].copy()


print(
    "Motor neurons         :",
    len(motor)
)


# ============================================================
# IDS
# ============================================================

sensory_ids = sensory[
    "bodyId"
].dropna().astype(np.int64).to_numpy()

dn_ids = descending[
    "bodyId"
].dropna().astype(np.int64).to_numpy()

motor_ids = motor[
    "bodyId"
].dropna().astype(np.int64).to_numpy()


# ============================================================
# LECTURE DU CONNECTOME
#
# On ne charge PAS le fichier de 1 Go intégralement.
# ============================================================

print()
print("Scan du connectome...")

source = pa.memory_map(
    str(CONNECTOME),
    "r"
)

reader = ipc.open_file(source)

selected_chunks = []


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
    # On ne garde que les arêtes potentiellement intéressantes :
    #
    # sensoriel ->
    # -> DN
    # DN ->
    # -> moteur
    # --------------------------------------------------------

    relevant = (
        (weight >= MIN_WEIGHT)
        &
        (
            np.isin(pre, sensory_ids)
            |
            np.isin(post, dn_ids)
            |
            np.isin(pre, dn_ids)
            |
            np.isin(post, motor_ids)
        )
    )


    if relevant.any():

        selected_chunks.append(
            pd.DataFrame(
                {
                    "body_pre":
                        pre[relevant],

                    "body_post":
                        post[relevant],

                    "weight":
                        weight[relevant],
                }
            )
        )


    if (
        i % 200 == 0
        or i == reader.num_record_batches - 1
    ):

        print(
            f"  batch "
            f"{i + 1}/"
            f"{reader.num_record_batches}"
        )


edges = pd.concat(
    selected_chunks,
    ignore_index=True
)

print()
print(
    "Connexions candidates :",
    len(edges)
)


# ============================================================
# SENSORIEL -> DN
# ============================================================

sensory_out = edges[
    edges["body_pre"].isin(
        sensory_ids
    )
]


dn_in = edges[
    edges["body_post"].isin(
        dn_ids
    )
]


# Connexions directes
direct_sensor_dn = sensory_out[
    sensory_out["body_post"].isin(
        dn_ids
    )
]


# ------------------------------------------------------------
# Connexions via UN neurone intermédiaire
#
# sensory -> X -> DN
# ------------------------------------------------------------

sensor_neighbors = set(
    sensory_out["body_post"]
)

dn_upstream = set(
    dn_in["body_pre"]
)


sensor_dn_intermediates = (
    sensor_neighbors
    & dn_upstream
)


# On ne considère pas DN eux-mêmes
sensor_dn_intermediates -= set(
    dn_ids
)


print()
print(
    "Intermediaires sensoriel -> DN :",
    len(sensor_dn_intermediates)
)


sensor_to_intermediate = sensory_out[
    sensory_out["body_post"].isin(
        sensor_dn_intermediates
    )
]


intermediate_to_dn = dn_in[
    dn_in["body_pre"].isin(
        sensor_dn_intermediates
    )
]


# DNs réellement atteints
connected_dn_ids = set(
    direct_sensor_dn[
        "body_post"
    ]
)

connected_dn_ids.update(
    intermediate_to_dn[
        "body_post"
    ]
)


print(
    "DN connectes au sensoriel      :",
    len(connected_dn_ids)
)


# ============================================================
# DN -> MOTOR
# ============================================================

dn_out = edges[
    edges["body_pre"].isin(
        connected_dn_ids
    )
]


motor_in = edges[
    edges["body_post"].isin(
        motor_ids
    )
]


# Direct
direct_dn_motor = dn_out[
    dn_out["body_post"].isin(
        motor_ids
    )
]


# ------------------------------------------------------------
# DN -> X -> MOTOR
# ------------------------------------------------------------

dn_neighbors = set(
    dn_out["body_post"]
)

motor_upstream = set(
    motor_in["body_pre"]
)


dn_motor_intermediates = (
    dn_neighbors
    & motor_upstream
)

dn_motor_intermediates -= set(
    connected_dn_ids
)


print(
    "Intermediaires DN -> moteur     :",
    len(dn_motor_intermediates)
)


dn_to_intermediate = dn_out[
    dn_out["body_post"].isin(
        dn_motor_intermediates
    )
]


intermediate_to_motor = motor_in[
    motor_in["body_pre"].isin(
        dn_motor_intermediates
    )
]


connected_motor_ids = set(
    direct_dn_motor[
        "body_post"
    ]
)

connected_motor_ids.update(
    intermediate_to_motor[
        "body_post"
    ]
)


print(
    "Motor neurons connectes        :",
    len(connected_motor_ids)
)


# ============================================================
# ASSEMBLAGE DU SOUS-RESEAU
# ============================================================

final_edges = pd.concat(
    [
        direct_sensor_dn,
        sensor_to_intermediate,
        intermediate_to_dn,

        direct_dn_motor,
        dn_to_intermediate,
        intermediate_to_motor,
    ],
    ignore_index=True,
)


final_edges = (
    final_edges
    .drop_duplicates()
    .reset_index(drop=True)
)


final_node_ids = set(
    final_edges["body_pre"]
)

final_node_ids.update(
    final_edges["body_post"]
)


nodes = ann[
    ann["bodyId"].isin(
        final_node_ids
    )
].copy()


# ============================================================
# AJOUT NEUROTRANSMETTEURS
# ============================================================

print()
print(
    "Chargement neurotransmetteurs..."
)

nt = feather.read_feather(
    NEUROTRANSMITTERS
)


# Il peut exister plusieurs lignes par body.
# On garde ici la prédiction la plus confiante.

nt_best = (
    nt
    .sort_values(
        "predicted_nt_confidence",
        ascending=False
    )
    .drop_duplicates(
        subset=["body"]
    )
)


nodes = nodes.merge(
    nt_best[
        [
            "body",
            "consensus_nt",
            "predicted_nt_confidence",
        ]
    ],
    left_on="bodyId",
    right_on="body",
    how="left",
)


# ============================================================
# ROLE DES NEURONES
# ============================================================

sensory_set = set(
    sensory_ids
)

dn_set = set(
    connected_dn_ids
)

motor_set = set(
    connected_motor_ids
)


def classify_role(body_id):

    if body_id in sensory_set:
        return "sensory"

    if body_id in dn_set:
        return "descending"

    if body_id in motor_set:
        return "motor"

    if body_id in sensor_dn_intermediates:
        return "sensory_to_dn"

    if body_id in dn_motor_intermediates:
        return "dn_to_motor"

    return "other"


nodes["neuroflight_role"] = (
    nodes["bodyId"]
    .apply(classify_role)
)


# ============================================================
# SAUVEGARDE
# ============================================================

OUT_DIR = DATA / "neuroflight"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


nodes_file = (
    OUT_DIR
    / "flight_subgraph_nodes.feather"
)

edges_file = (
    OUT_DIR
    / "flight_subgraph_edges.feather"
)


nodes.reset_index(
    drop=True
).to_feather(
    nodes_file
)


final_edges.reset_index(
    drop=True
).to_feather(
    edges_file
)


# Versions CSV lisibles
nodes.to_csv(
    OUT_DIR
    / "flight_subgraph_nodes.csv",
    index=False
)

final_edges.to_csv(
    OUT_DIR
    / "flight_subgraph_edges.csv",
    index=False
)


# ============================================================
# RESUME
# ============================================================

print()
print("=" * 70)
print("NEUROFLIGHT FLY SUBGRAPH")
print("=" * 70)

print(
    "Neurones sensoriels       :",
    sum(
        nodes["neuroflight_role"]
        == "sensory"
    )
)

print(
    "Intermediaires sensor->DN :",
    sum(
        nodes["neuroflight_role"]
        == "sensory_to_dn"
    )
)

print(
    "Descending neurons        :",
    sum(
        nodes["neuroflight_role"]
        == "descending"
    )
)

print(
    "Intermediaires DN->motor  :",
    sum(
        nodes["neuroflight_role"]
        == "dn_to_motor"
    )
)

print(
    "Motor neurons             :",
    sum(
        nodes["neuroflight_role"]
        == "motor"
    )
)

print()
print(
    "TOTAL neurones :",
    len(nodes)
)

print(
    "TOTAL connexions :",
    len(final_edges)
)


print()
print("Fichiers crees :")

print(nodes_file)
print(edges_file)


# ============================================================
# TOP DN
# ============================================================

dn_inputs = final_edges[
    final_edges["body_post"].isin(
        connected_dn_ids
    )
]


scores = (
    dn_inputs
    .groupby("body_post")["weight"]
    .sum()
    .sort_values(
        ascending=False
    )
)


top_dn = (
    nodes[
        nodes["bodyId"].isin(
            scores.index
        )
    ]
    .copy()
)


top_dn["input_weight"] = (
    top_dn["bodyId"]
    .map(scores)
)


top_dn = top_dn.sort_values(
    "input_weight",
    ascending=False
)


print()
print("=" * 70)
print("TOP DESCENDING NEURONS")
print("=" * 70)

print(
    top_dn[
        [
            "bodyId",
            "type",
            "instance",
            "consensus_nt",
            "input_weight",
        ]
    ]
    .head(30)
    .to_string(index=False)
)