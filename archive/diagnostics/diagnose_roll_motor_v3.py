from pathlib import Path
import csv
import math

import mujoco
import numpy as np

from scripts.test_malecns_roll_reflex import (
    brain,
    metadata,
    model,
    get_gyro,
    haltere_left,
    haltere_right,
    BASE_HALTERE_RATE,
    GYRO_TO_HZ,
    MAX_DIFFERENTIAL_HZ,
    MOTOR_TRACE_TAU_MS,
    MOTOR_GAIN,
)


# ============================================================
# CONFIG
# ============================================================

DURATION_SECONDS = 5.0

INITIAL_ROLL_RATES = [
    +1.5,
    -1.5,
]

SEED = 42

OUTPUT_DIR = Path(
    "logs/roll_diagnostic"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
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

        if value is None:
            continue

        if str(value) == "nan":
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

    instance = (
        str(instance)
        .strip()
        .upper()
    )


    if instance.endswith("_L"):
        return "L"

    if instance.endswith("_R"):
        return "R"


    return "?"


# ============================================================
# IDENTIFICATION DES 10 MN DE VOL
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


FLIGHT_TYPES = [
    "b1",
    "b2",
    "b3",
    "i1",
    "i2",
]


motor_indices = {}


for motor_type in FLIGHT_TYPES:

    target = (
        motor_type
        + "mn"
    )


    indices = np.flatnonzero(
        (
            type_clean
            == target
        ).to_numpy()
    )


    for index in indices:

        side = infer_side(
            metadata.iloc[
                index
            ]
        )


        if side in (
            "L",
            "R",
        ):

            key = (
                motor_type
                + "_"
                + side
            )

            motor_indices[
                key
            ] = int(
                index
            )


print()
print("=" * 60)
print("FLIGHT MOTOR NEURONS")
print("=" * 60)

for key in sorted(
    motor_indices
):

    index = motor_indices[
        key
    ]

    row = metadata.iloc[
        index
    ]

    print(
        f"{key:5s} | "
        f"body={row['bodyId']} | "
        f"type={row['type']}"
    )


# ============================================================
# LOOKUP RAPIDE
# ============================================================

motor_lookup = {}


for key, index in (
    motor_indices.items()
):

    motor_lookup[
        index
    ] = key


# ============================================================
# PHYSICS
# ============================================================

physics_dt = float(
    model.opt.timestep
)

physics_dt_ms = (
    physics_dt
    * 1000.0
)


lif_steps_per_physics = max(
    1,
    int(
        round(
            physics_dt_ms
            /
            brain.dt_ms
        )
    ),
)


# ============================================================
# EXPERIENCE
# ============================================================

def run_diagnostic(
    initial_roll_rate,
    seed,
):

    direction = (
        "positive"
        if initial_roll_rate > 0
        else "negative"
    )


    output_file = (
        OUTPUT_DIR
        /
        f"roll_motor_{direction}_seed{seed}.csv"
    )


    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    data.qpos[
        0:3
    ] = [
        0.0,
        0.0,
        0.5,
    ]


    data.qpos[
        3:7
    ] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


    data.qvel[:] = 0.0

    data.qvel[3] = (
        initial_roll_rate
    )


    data.ctrl[:] = 0.0


    mujoco.mj_forward(
        model,
        data,
    )


    brain.reset()


    rng = np.random.default_rng(
        seed
    )


    # ========================================================
    # TRACE B1/B2 UTILISEE ACTUELLEMENT
    # ========================================================

    left_trace = 0.0
    right_trace = 0.0


    trace_decay = math.exp(
        -physics_dt_ms
        /
        MOTOR_TRACE_TAU_MS
    )


    # ========================================================
    # COMPTEURS PAR FENETRE DE 100 ms
    # ========================================================

    window_counts = {
        key: 0
        for key
        in motor_indices
    }


    window_forced_left = 0
    window_forced_right = 0


    total_steps = int(
        DURATION_SECONDS
        /
        physics_dt
    )


    log_steps = max(
        1,
        int(
            round(
                0.1
                /
                physics_dt
            )
        ),
    )


    rows = []


    # ========================================================
    # LOOP
    # ========================================================

    for step in range(
        total_steps
    ):

        gyro_x = float(
            get_gyro(
                data
            )[0]
        )


        differential = np.clip(
            GYRO_TO_HZ
            * gyro_x,

            -MAX_DIFFERENTIAL_HZ,
            MAX_DIFFERENTIAL_HZ,
        )


        left_rate = np.clip(
            BASE_HALTERE_RATE
            + differential,

            0.0,
            150.0,
        )


        right_rate = np.clip(
            BASE_HALTERE_RATE
            - differential,

            0.0,
            150.0,
        )


        # ====================================================
        # CERVEAU
        # ====================================================

        b12_left_spikes = 0
        b12_right_spikes = 0


        for _ in range(
            lif_steps_per_physics
        ):

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


            forced_left = (
                haltere_left[
                    rng.random(
                        len(
                            haltere_left
                        )
                    )
                    < p_left
                ]
            )


            forced_right = (
                haltere_right[
                    rng.random(
                        len(
                            haltere_right
                        )
                    )
                    < p_right
                ]
            )


            window_forced_left += len(
                forced_left
            )

            window_forced_right += len(
                forced_right
            )


            forced = np.concatenate(
                [
                    forced_left,
                    forced_right,
                ]
            )


            spikes, _ = brain.step(
                forced_spikes=forced
            )


            # ================================================
            # COMPTAGE DES 10 MOTOR NEURONS
            # ================================================

            for spike in spikes:

                key = motor_lookup.get(
                    int(spike)
                )

                if key is None:
                    continue


                window_counts[
                    key
                ] += 1


                if key in (
                    "b1_L",
                    "b2_L",
                ):

                    b12_left_spikes += 1


                elif key in (
                    "b1_R",
                    "b2_R",
                ):

                    b12_right_spikes += 1


        # ====================================================
        # DECODEUR ACTUEL
        # ====================================================

        left_trace = (
            left_trace
            * trace_decay
            +
            b12_left_spikes
        )


        right_trace = (
            right_trace
            * trace_decay
            +
            b12_right_spikes
        )


        denominator = (
            left_trace
            +
            right_trace
            +
            1e-6
        )


        brain_signal = (
            left_trace
            -
            right_trace
        ) / denominator


        control = np.clip(
            MOTOR_GAIN
            * brain_signal,

            -1.0,
            1.0,
        )


        # ====================================================
        # CRAZYFLIE
        # ====================================================

        data.ctrl[:] = 0.0

        data.ctrl[1] = (
            control
        )


        mujoco.mj_step(
            model,
            data,
        )


        # ====================================================
        # LOG 100 ms
        # ====================================================

        if (
            (step + 1)
            % log_steps
            == 0
        ):

            time_s = (
                (step + 1)
                * physics_dt
            )


            row = {

                "time":
                    time_s,

                "gyro":
                    gyro_x,

                "haltere_rate_L":
                    float(
                        left_rate
                    ),

                "haltere_rate_R":
                    float(
                        right_rate
                    ),

                "haltere_spikes_L":
                    window_forced_left,

                "haltere_spikes_R":
                    window_forced_right,

                "brain_signal":
                    float(
                        brain_signal
                    ),

                "control":
                    float(
                        control
                    ),
            }


            for key in sorted(
                motor_indices
            ):

                row[
                    key
                ] = (
                    window_counts[
                        key
                    ]
                )


            rows.append(
                row
            )


            # ================================================
            # RESET FENETRE
            # ================================================

            for key in (
                window_counts
            ):

                window_counts[
                    key
                ] = 0


            window_forced_left = 0
            window_forced_right = 0


    # ========================================================
    # CSV
    # ========================================================

    fieldnames = [
        "time",
        "gyro",

        "haltere_rate_L",
        "haltere_rate_R",

        "haltere_spikes_L",
        "haltere_spikes_R",

        "brain_signal",
        "control",
    ]


    fieldnames.extend(
        sorted(
            motor_indices
        )
    )


    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    final_gyro = float(
        get_gyro(
            data
        )[0]
    )


    print()
    print(
        direction.upper()
    )

    print(
        "Initial :",
        f"{initial_roll_rate:+.4f}"
    )

    print(
        "Final   :",
        f"{final_gyro:+.4f}"
    )

    print(
        "CSV     :",
        output_file
    )


# ============================================================
# RUN
# ============================================================

for initial_rate in (
    INITIAL_ROLL_RATES
):

    run_diagnostic(
        initial_roll_rate=
            initial_rate,

        seed=SEED,
    )