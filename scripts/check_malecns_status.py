from pathlib import Path
import pandas as pd
import pyarrow.feather as feather


DATA = Path("data/male_cns")

ANNOTATIONS = DATA / (
    "body-annotations-male-cns-v1.0-minconf-0.5.feather"
)


ann = feather.read_feather(
    ANNOTATIONS
)


print()
print("=" * 60)
print("MALE CNS ANNOTATIONS")
print("=" * 60)

print(
    "Total lignes :",
    len(ann)
)

print(
    "BodyId uniques :",
    ann["bodyId"].nunique()
)


print()
print("=" * 60)
print("STATUS")
print("=" * 60)

if "status" in ann.columns:

    print(
        ann["status"]
        .fillna("<EMPTY>")
        .value_counts(
            dropna=False
        )
        .head(30)
    )


print()
print("=" * 60)
print("STATUS LABEL")
print("=" * 60)

if "statusLabel" in ann.columns:

    print(
        ann["statusLabel"]
        .astype(str)
        .fillna("<EMPTY>")
        .value_counts(
            dropna=False
        )
        .head(30)
    )


print()
print("=" * 60)
print("SUPERCLASS")
print("=" * 60)

print(
    ann["superclass"]
    .fillna("<EMPTY>")
    .value_counts()
    .head(30)
)