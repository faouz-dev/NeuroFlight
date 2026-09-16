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
    MOTOR_GAIN,
)


# ============================================================
# CONFIG
# ============================================================

SIMULATION_SECONDS = 5.0

INITIAL_ROLL_RATE = 1.5

FAST_TAU_MS = 50.0
SLOW_TAU_MS = 120.0

# Importance du canal b3/i2
SLOW_GAIN = 2.0


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

        value = str(
            value
        ).strip().upper()

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


    instance = str(
        row.get(
            "instance",
            ""
        )
    ).strip().upper()


    if instance.endswith("_L"):
        return "L"

    if instance.endswith("_R"):
        return "R"


    return "?"


# ============================================================
# MOTOR POPULATIONS
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


def find_motor(
    motor_name,
    side,
):

    indices = np.flatnonzero(
        (
            type_clean
            == (
                motor_name
                + "mn"
            )
        ).to_numpy()
    )


    result = []


    for index in indices:

        if infer_side(
            metadata.iloc[
                index
            ]
        ) == side:

            result.append(
                index
            )


    return np.asarray(
        result,
        dtype=np.int64,
    )


# b1 / b2
b1_L = find_motor(
    "b1",
    "L",
)

b1_R = find_motor(
    "b1",
    "R",
)

b2_L = find_motor(
    "b2",
    "L",
)

b2_R = find_motor(
    "b2",
    "R",
)


# b3
b3_L = find_motor(
    "b3",
    "L",
)

b3_R = find_motor(
    "b3",
    "R",
)


# i2
i2_L = find_motor(
    "i2",
    "L",
)

i2_R = find_motor(
    "i2",
    "R",
)


print()
print("=" * 70)
print("ROLL DECODER V2")
print("=" * 70)

print(
    "b1 L/R :",
    len(b1_L),
    "/",
    len(b1_R)
)

print(
    "b2 L/R :",
    len(b2_L),
    "/",
    len(b2_R)
)

print(
    "b3 L/R :",
    len(b3_L),
    "/",
    len(b3_R)
)

print(
    "i2 L/R :",
    len(i2_L),
    "/",
    len(i2_R)
)


# ============================================================
# LOOKUP MASKS
# ============================================================

def make_mask(indices):

    mask = np.zeros(
        brain.num_neurons,
        dtype=bool,
    )

    mask[
        indices
    ] = True

    return mask


b12_L_mask = make_mask(
    np.concatenate(
        [
            b1_L,
            b2_L,
        ]
    )
)

b12_R_mask = make_mask(
    np.concatenate(
        [
            b1_R,
            b2_R,
        ]
    )
)


b3_L_mask = make_mask(
    b3_L
)

b3_R_mask = make_mask(
    b3_R
)


i2_L_mask = make_mask(
    i2_L
)

i2_R_mask = make_mask(
    i2_R
)


# ============================================================
# MUJOCO TIMING
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


fast_decay = math.exp(
    -physics_dt_ms
    /
    FAST_TAU_MS
)


slow_decay = math.exp(
    -physics_dt_ms
    /
    SLOW_TAU_MS
)


# ============================================================
# RUN
# ============================================================

def run(
    initial_roll_rate,
    seed,
):

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
    # TRACES B1/B2
    # ========================================================

    fast_L = 0.0
    fast_R = 0.0


    # ========================================================
    # TRACES B3 / I2
    # ========================================================

    b3L_trace = 0.0
    b3R_trace = 0.0

    i2L_trace = 0.0
    i2R_trace = 0.0


    total_steps = int(
        SIMULATION_SECONDS
        /
        physics_dt
    )


    log_interval = max(
        1,
        int(
            0.1
            /
            physics_dt
        ),
    )


    print()
    print(
        "Initial gyro:",
        f"{initial_roll_rate:+.3f}"
    )


    for step in range(
        total_steps
    ):

        gyro_x = float(
            get_gyro(
                data
            )[0]
        )


        # ====================================================
        # GYRO -> HALTERES
        # ====================================================

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
        # SPIKE COUNTS POUR CE PAS PHYSIQUE
        # ====================================================

        fastL_spikes = 0
        fastR_spikes = 0

        b3L_spikes = 0
        b3R_spikes = 0

        i2L_spikes = 0
        i2R_spikes = 0


        # ====================================================
        # LIF
        # ====================================================

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


            forced = np.concatenate(
                [
                    forced_left,
                    forced_right,
                ]
            )


            spikes, _ = brain.step(
                forced_spikes=forced
            )


            if not len(
                spikes
            ):
                continue


            fastL_spikes += int(
                b12_L_mask[
                    spikes
                ].sum()
            )

            fastR_spikes += int(
                b12_R_mask[
                    spikes
                ].sum()
            )


            b3L_spikes += int(
                b3_L_mask[
                    spikes
                ].sum()
            )

            b3R_spikes += int(
                b3_R_mask[
                    spikes
                ].sum()
            )


            i2L_spikes += int(
                i2_L_mask[
                    spikes
                ].sum()
            )

            i2R_spikes += int(
                i2_R_mask[
                    spikes
                ].sum()
            )


        # ====================================================
        # UPDATE TRACES
        # ====================================================

        fast_L = (
            fast_L
            * fast_decay
            +
            fastL_spikes
        )


        fast_R = (
            fast_R
            * fast_decay
            +
            fastR_spikes
        )


        b3L_trace = (
            b3L_trace
            * slow_decay
            +
            b3L_spikes
        )


        b3R_trace = (
            b3R_trace
            * slow_decay
            +
            b3R_spikes
        )


        i2L_trace = (
            i2L_trace
            * slow_decay
            +
            i2L_spikes
        )


        i2R_trace = (
            i2R_trace
            * slow_decay
            +
            i2R_spikes
        )


        # ====================================================
        # FAST SIGNAL : B1/B2
        # ====================================================

        fast_denominator = (
            fast_L
            +
            fast_R
            +
            1e-6
        )


        fast_signal = (
            fast_L
            -
            fast_R
        ) / fast_denominator


        # ====================================================
        # SLOW SIGNAL : B3 - I2
        #
        # d_b3 = b3L - b3R
        # d_i2 = i2L - i2R
        #
        # slow = d_b3 - d_i2
        # ====================================================

        slow_numerator = (

            (
                b3L_trace
                -
                b3R_trace
            )

            -

            (
                i2L_trace
                -
                i2R_trace
            )
        )


        slow_denominator = (
            b3L_trace
            +
            b3R_trace
            +
            i2L_trace
            +
            i2R_trace
            +
            1e-6
        )


        slow_signal = (
            slow_numerator
            /
            slow_denominator
        )


        # ====================================================
        # COMBINAISON
        #
        # B1/B2 reste le canal rapide principal.
        #
        # B3/I2 remplit les longues périodes silencieuses.
        # ====================================================

        brain_signal = np.clip(

            fast_signal
            +
            SLOW_GAIN
            * slow_signal,

            -1.0,
            1.0,
        )


        # ====================================================
        # CRAZYFLIE
        # ====================================================

        control = np.clip(

            MOTOR_GAIN
            * brain_signal,

            -1.0,
            1.0,
        )


        data.ctrl[:] = 0.0

        data.ctrl[1] = (
            control
        )


        mujoco.mj_step(
            model,
            data,
        )


        # ====================================================
        # LOG
        # ====================================================

        if (
            step
            % log_interval
            == 0
        ):

            print(

                f"t="
                f"{step * physics_dt:4.2f}s | "

                f"gyro="
                f"{gyro_x:+6.3f} | "

                f"fast="
                f"{fast_signal:+6.3f} | "

                f"slow="
                f"{slow_signal:+6.3f} | "

                f"brain="
                f"{brain_signal:+6.3f} | "

                f"ctrl="
                f"{control:+6.3f}"
            )


    final_gyro = float(
        get_gyro(
            data
        )[0]
    )


    print()

    print(
        "Final gyro:",
        f"{final_gyro:+.4f}"
    )


    return final_gyro


# ============================================================
# POSITIVE
# ============================================================

positive = run(
    initial_roll_rate=
        INITIAL_ROLL_RATE,

    seed=42,
)


# ============================================================
# NEGATIVE
# ============================================================

negative = run(
    initial_roll_rate=
        -INITIAL_ROLL_RATE,

    seed=42,
)


print()
print("=" * 70)
print("DECODER V2 VERDICT")
print("=" * 70)

print(
    "Positive :",
    f"{positive:+.4f}"
)

print(
    "Negative :",
    f"{negative:+.4f}"
)