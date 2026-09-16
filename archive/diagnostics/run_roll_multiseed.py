from pathlib import Path
from datetime import datetime
import contextlib
import io
import csv
import statistics

from scripts.test_malecns_roll_reflex import run


# ============================================================
# CONFIG
# ============================================================

N_SEEDS = 10

INITIAL_RATE = 1.5

OUTPUT_DIR = Path("logs/roll_multiseed")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)


FULL_LOG = (
    OUTPUT_DIR
    / f"roll_multiseed_{timestamp}.txt"
)

CSV_FILE = (
    OUTPUT_DIR
    / f"roll_multiseed_{timestamp}.csv"
)


# ============================================================
# RESULTATS
# ============================================================

results = []


def log(text=""):

    print(text)

    with open(
        FULL_LOG,
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            str(text)
            + "\n"
        )


# ============================================================
# HEADER
# ============================================================

log("=" * 70)
log("NEUROFLIGHT - MALECNS ROLL MULTI-SEED")
log("=" * 70)

log(
    f"Nombre de seeds : {N_SEEDS}"
)

log(
    f"Initial rate    : +/-{INITIAL_RATE} rad/s"
)

log()


# ============================================================
# TESTS
# ============================================================

for seed in range(
    1,
    N_SEEDS + 1,
):

    log()
    log("=" * 70)

    log(
        f"SEED {seed}/{N_SEEDS}"
    )

    log("=" * 70)


    # --------------------------------------------------------
    # POSITIF
    # --------------------------------------------------------

    buffer = io.StringIO()


    with contextlib.redirect_stdout(
        buffer
    ):

        final_positive = run(
            initial_roll_rate=INITIAL_RATE,
            use_brain=True,
            seed=seed,
        )


    positive_log = (
        buffer.getvalue()
    )


    log(
        positive_log.rstrip()
    )


    # --------------------------------------------------------
    # NEGATIF
    # --------------------------------------------------------

    buffer = io.StringIO()


    with contextlib.redirect_stdout(
        buffer
    ):

        final_negative = run(
            initial_roll_rate=-INITIAL_RATE,
            use_brain=True,
            seed=(
                1000 + seed
            ),
        )


    negative_log = (
        buffer.getvalue()
    )


    log(
        negative_log.rstrip()
    )


    # --------------------------------------------------------
    # STOCKAGE
    # --------------------------------------------------------

    reduction_positive = (
        1.0
        -
        abs(final_positive)
        / INITIAL_RATE
    ) * 100.0


    reduction_negative = (
        1.0
        -
        abs(final_negative)
        / INITIAL_RATE
    ) * 100.0


    results.append(
        {
            "seed": seed,

            "initial_positive":
                INITIAL_RATE,

            "final_positive":
                final_positive,

            "reduction_positive_percent":
                reduction_positive,

            "initial_negative":
                -INITIAL_RATE,

            "final_negative":
                final_negative,

            "reduction_negative_percent":
                reduction_negative,
        }
    )


    log()
    log(
        f"RESUME SEED {seed}"
    )

    log(
        f"  +{INITIAL_RATE:.1f}"
        f" -> {final_positive:+.4f}"
        f" | reduction "
        f"{reduction_positive:.1f}%"
    )

    log(
        f"  -{INITIAL_RATE:.1f}"
        f" -> {final_negative:+.4f}"
        f" | reduction "
        f"{reduction_negative:.1f}%"
    )


# ============================================================
# CSV
# ============================================================

with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "seed",

            "initial_positive",
            "final_positive",
            "reduction_positive_percent",

            "initial_negative",
            "final_negative",
            "reduction_negative_percent",
        ],
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# STATISTIQUES
# ============================================================

positive_values = [
    result[
        "final_positive"
    ]
    for result in results
]


negative_values = [
    result[
        "final_negative"
    ]
    for result in results
]


positive_reduction = [
    result[
        "reduction_positive_percent"
    ]
    for result in results
]


negative_reduction = [
    result[
        "reduction_negative_percent"
    ]
    for result in results
]


log()
log()
log("=" * 70)
log("RESULTATS FINAUX")
log("=" * 70)


log()

log("ROLL POSITIF")

log(
    "  moyenne finale : "
    f"{statistics.mean(positive_values):+.4f}"
)

log(
    "  min            : "
    f"{min(positive_values):+.4f}"
)

log(
    "  max            : "
    f"{max(positive_values):+.4f}"
)

log(
    "  reduction moy. : "
    f"{statistics.mean(positive_reduction):.1f}%"
)


log()

log("ROLL NEGATIF")

log(
    "  moyenne finale : "
    f"{statistics.mean(negative_values):+.4f}"
)

log(
    "  min            : "
    f"{min(negative_values):+.4f}"
)

log(
    "  max            : "
    f"{max(negative_values):+.4f}"
)

log(
    "  reduction moy. : "
    f"{statistics.mean(negative_reduction):.1f}%"
)


# ============================================================
# ASYMETRIE
# ============================================================

positive_abs = statistics.mean(
    abs(value)
    for value in positive_values
)

negative_abs = statistics.mean(
    abs(value)
    for value in negative_values
)


asymmetry = abs(
    positive_abs
    - negative_abs
)


log()

log(
    "Asymetrie moyenne |+/-| : "
    f"{asymmetry:.4f} rad/s"
)


# ============================================================
# FILES
# ============================================================

log()

log(
    f"Log complet : {FULL_LOG}"
)

log(
    f"CSV         : {CSV_FILE}"
)