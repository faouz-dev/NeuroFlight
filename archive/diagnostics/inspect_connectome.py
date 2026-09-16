from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.feather as feather


DATA_DIR = Path("data/male_cns")

FILES = {
    "annotations": DATA_DIR / (
        "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    ),

    "neurotransmitters": DATA_DIR / (
        "body-neurotransmitters-male-cns-v1.0.feather"
    ),

    "connectome": DATA_DIR / (
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
    ),
}


def print_file_info(name, path):

    print()
    print("=" * 70)
    print(name.upper())
    print("=" * 70)

    if not path.exists():
        print("FICHIER INTROUVABLE :", path)
        return

    size_mb = path.stat().st_size / (1024 * 1024)

    print("Fichier :", path)
    print(f"Taille  : {size_mb:.2f} MB")

    try:
        # Feather V2 utilise le format Arrow IPC.
        # Memory map permet d'éviter de charger tout le gros fichier.
        source = pa.memory_map(str(path), "r")
        reader = ipc.open_file(source)

        print()
        print("Nombre de blocs :", reader.num_record_batches)

        print()
        print("Colonnes / schema :")
        print(reader.schema)

        if reader.num_record_batches > 0:

            batch = reader.get_batch(0)

            sample = batch.slice(
                0,
                min(5, batch.num_rows)
            )

            print()
            print("Exemple des premières lignes :")
            print(sample.to_pandas())

    except Exception as e:

        print()
        print("Lecture IPC impossible :", e)
        print("Essai avec pyarrow.feather...")

        table = feather.read_table(path)

        print()
        print("Schema :")
        print(table.schema)

        print()
        print("Exemple :")
        print(
            table.slice(
                0,
                min(5, table.num_rows)
            ).to_pandas()
        )


for name, path in FILES.items():
    print_file_info(name, path)