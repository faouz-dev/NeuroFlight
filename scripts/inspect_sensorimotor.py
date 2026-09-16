from pathlib import Path

import pandas as pd
import pyarrow.feather as feather


DATA = Path("data/male_cns")

ANNOTATIONS = DATA / (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)

NT_FILE = DATA / (
    "body-neurotransmitters-male-cns-v1.0.feather"
)


# ============================================================
# CHARGEMENT
# ============================================================

print("Chargement des annotations...")

ann = feather.read_feather(ANNOTATIONS)

print("Neurones :", len(ann))


print()
print("Chargement neurotransmetteurs...")

nt = feather.read_feather(NT_FILE)

print("Entrees neurotransmetteurs :", len(nt))


# ============================================================
# DISTRIBUTION GENERALE
# ============================================================

columns_to_inspect = [
    "superclass",
    "class",
    "subclass",
    "somaNeuromere",
    "entryNerve",
    "exitNerve",
]


for column in columns_to_inspect:

    print()
    print("=" * 70)
    print(column.upper())
    print("=" * 70)

    if column not in ann.columns:
        continue

    values = (
        ann[column]
        .dropna()
        .astype(str)
        .value_counts()
        .head(40)
    )

    print(values)


# ============================================================
# DESCENDING NEURONS
# ============================================================

types = ann["type"].fillna("").astype(str)

descending = ann[
    types.str.startswith("DN", na=False)
].copy()


print()
print("=" * 70)
print("DESCENDING NEURONS")
print("=" * 70)

print("Nombre de candidats :", len(descending))

print(
    descending[
        [
            "bodyId",
            "type",
            "instance",
            "class",
            "subclass",
            "superclass",
            "somaNeuromere",
            "exitNerve",
        ]
    ].head(100).to_string(index=False)
)


# ============================================================
# RECHERCHE SENSORIMOTRICE
# ============================================================

search_columns = [
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "synonyms",
]


terms = [
    "visual",
    "optic",
    "motion",
    "motor",
    "premotor",
    "sensory",
    "proprio",
    "haltere",
]


combined = pd.Series(
    "",
    index=ann.index,
    dtype="object",
)

for column in search_columns:

    if column in ann.columns:

        combined += (
            " "
            + ann[column]
            .fillna("")
            .astype(str)
            .str.lower()
        )


mask = False

for term in terms:
    mask = mask | combined.str.contains(
        term,
        regex=False,
    )


sensorimotor = ann[mask].copy()


print()
print("=" * 70)
print("SENSORIMOTOR KEYWORD CANDIDATES")
print("=" * 70)

print(
    "Nombre de candidats :",
    len(sensorimotor)
)


print(
    sensorimotor[
        [
            "bodyId",
            "type",
            "instance",
            "class",
            "subclass",
            "superclass",
        ]
    ].head(150).to_string(index=False)
)


# ============================================================
# TYPES VISUELS POTENTIELLEMENT INTERESSANTS
# ============================================================

special = ann[
    types.str.startswith(
        (
            "VS",
            "HS",
            "DN",
        ),
        na=False,
    )
].copy()


print()
print("=" * 70)
print("DN / VS / HS")
print("=" * 70)

print(
    special[
        [
            "bodyId",
            "type",
            "instance",
            "class",
            "subclass",
            "superclass",
        ]
    ].head(200).to_string(index=False)
)


# ============================================================
# JOINDRE LES NEUROTRANSMETTEURS AUX DN
# ============================================================

dn_with_nt = descending.merge(
    nt[
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


print()
print("=" * 70)
print("DESCENDING NEURONS + NEUROTRANSMITTER")
print("=" * 70)

print(
    dn_with_nt[
        [
            "bodyId",
            "type",
            "consensus_nt",
            "predicted_nt_confidence",
        ]
    ].head(100).to_string(index=False)
)