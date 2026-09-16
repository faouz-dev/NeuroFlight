import math

import mujoco
import mujoco_menagerie as mm
import numpy as np
import pandas as pd

from flybrain.lif import MaleCNSLIF


# ============================================================
# CONFIG
# ============================================================

CACHE = "data/male_cns/cache_v3"

INITIAL_ROLL_RATE = 1.5       # rad/s

SIMULATION_SECONDS = 5.0

BASE_HALTERE_RATE = 75.0      # Hz

# gyro rad/s -> différence de fréquence haltère
GYRO_TO_HZ = 30.0

MAX_DIFFERENTIAL_HZ = 50.0

# Filtre temporel de la sortie motrice
MOTOR_TRACE_TAU_MS = 50.0

# Intensité maximum du moment Crazyflie
MOTOR_GAIN = 0.8


# ============================================================
# MALE CNS
# ============================================================

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


    instance = row.get(
        "instance",
        "",
    )

    if pd.notna(instance):

        instance = str(
            instance
        ).strip().upper()

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


haltere_sides = np.asarray(
    [
        infer_side(
            metadata.iloc[i]
        )
        for i in haltere_indices
    ]
)


haltere_left = haltere_indices[
    haltere_sides == "L"
]

haltere_right = haltere_indices[
    haltere_sides == "R"
]


# ============================================================
# b1 / b2 MOTOR NEURONS
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


steering_mask = (
    type_clean.isin(
        [
            "b1mn",
            "b2mn",
        ]
    )
    .to_numpy()
)


steering_indices = np.flatnonzero(
    steering_mask
)


steering_sides = np.asarray(
    [
        infer_side(
            metadata.iloc[i]
        )
        for i in steering_indices
    ]
)


motor_left = steering_indices[
    steering_sides == "L"
]

motor_right = steering_indices[
    steering_sides == "R"
]


print()
print("=" * 60)
print("MALE CNS ROLL REFLEX")
print("=" * 60)

print(
    "Haltere L :",
    len(haltere_left)
)

print(
    "Haltere R :",
    len(haltere_right)
)

print(
    "b1/b2 L   :",
    len(motor_left)
)

print(
    "b1/b2 R   :",
    len(motor_right)
)


# Lookup rapide
is_motor_left = np.zeros(
    brain.num_neurons,
    dtype=bool,
)

is_motor_right = np.zeros(
    brain.num_neurons,
    dtype=bool,
)

is_motor_left[
    motor_left
] = True

is_motor_right[
    motor_right
] = True


# ============================================================
# CRAZYFLIE
# ============================================================

model = mm.load(
    "bitcraze_crazyflie_2"
)


# Pour ce premier test :
# on isole uniquement la rotation.
model.opt.gravity[:] = 0.0


gyro_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SENSOR,
    "body_gyro",
)


gyro_address = int(
    model.sensor_adr[
        gyro_id
    ]
)


def get_gyro(data):

    return data.sensordata[
        gyro_address:
        gyro_address + 3
    ].copy()


physics_dt = float(
    model.opt.timestep
)

physics_dt_ms = (
    physics_dt * 1000.0
)


lif_steps_per_physics = max(
    1,
    int(
        round(
            physics_dt_ms
            / brain.dt_ms
        )
    ),
)


print(
    "MuJoCo timestep :",
    physics_dt_ms,
    "ms"
)

print(
    "LIF steps / physics step :",
    lif_steps_per_physics
)


# ============================================================
# SIMULATION
# ============================================================

def run(
    initial_roll_rate,
    use_brain,
    seed,
):

    data = mujoco.MjData(
        model
    )


    mujoco.mj_resetData(
        model,
        data,
    )


    # Position arbitraire
    data.qpos[0:3] = [
        0.0,
        0.0,
        0.5,
    ]


    # Orientation initiale horizontale
    data.qpos[3:7] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]


    # Vitesse angulaire X initiale
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


    left_trace = 0.0
    right_trace = 0.0


    trace_decay = math.exp(
        -physics_dt_ms
        / MOTOR_TRACE_TAU_MS
    )


    total_steps = int(
        SIMULATION_SECONDS
        / physics_dt
    )


    log_interval = max(
        1,
        int(
            0.1
            / physics_dt
        ),
    )


    print()
    print(
        "BRAIN"
        if use_brain
        else "BASELINE"
    )

    print(
        "Initial gyro X :",
        initial_roll_rate,
    )


    for step in range(
        total_steps
    ):

        gyro = get_gyro(
            data
        )


        gyro_x = float(
            gyro[0]
        )


        # ====================================================
        # GYRO -> HALTERE
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


        left_spikes = 0
        right_spikes = 0


        # ====================================================
        # MALE CNS
        # ====================================================

        if use_brain:

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


                if len(spikes):

                    left_spikes += int(
                        is_motor_left[
                            spikes
                        ].sum()
                    )

                    right_spikes += int(
                        is_motor_right[
                            spikes
                        ].sum()
                    )


            # =================================================
            # FILTRE MOTOR
            # =================================================

            left_trace = (
                left_trace
                * trace_decay
                + left_spikes
            )


            right_trace = (
                right_trace
                * trace_decay
                + right_spikes
            )


            # Signal normalisé [-1, +1]
            denominator = (
                left_trace
                + right_trace
                + 1e-6
            )


            roll_brain = (
                left_trace
                - right_trace
            ) / denominator


            # =================================================
            # ADAPTATEUR MOUCHE -> QUADROTOR
            #
            # Ce gain ne représente PAS un muscle biologique.
            # Il traduit simplement l'asymétrie des ailes
            # vers l'actionneur X du Crazyflie.
            # =================================================

            x_command = np.clip(
                MOTOR_GAIN
                * roll_brain,
                -1.0,
                1.0,
            )

        else:

            roll_brain = 0.0
            x_command = 0.0


        # ====================================================
        # MUJOCO
        # ====================================================

        data.ctrl[:] = 0.0

        # Actuator x_moment
        data.ctrl[1] = (
            x_command
        )


        mujoco.mj_step(
            model,
            data
        )


        if (
            step % log_interval
            == 0
        ):

            print(
                f"t="
                f"{step * physics_dt:4.2f}s | "
                f"gyro={gyro_x:+6.3f} | "
                f"rates="
                f"{left_rate:5.1f}/"
                f"{right_rate:5.1f} | "
                f"brain="
                f"{roll_brain:+6.3f} | "
                f"ctrl="
                f"{x_command:+6.3f}"
            )


    final_gyro = float(
        get_gyro(
            data
        )[0]
    )


    print(
        "Final gyro X :",
        f"{final_gyro:+.4f}"
    )


    return final_gyro


# ============================================================
# 1. BASELINE
# ============================================================
if __name__ == "__main__":

    # ========================================================
    # 1. BASELINE
    # ========================================================

    baseline = run(
        initial_roll_rate=INITIAL_ROLL_RATE,
        use_brain=False,
        seed=1,
    )


    # ========================================================
    # 2. MALE CNS POSITIF
    # ========================================================

    brain_positive = run(
        initial_roll_rate=INITIAL_ROLL_RATE,
        use_brain=True,
        seed=42,
    )


    # ========================================================
    # 3. MALE CNS NEGATIF
    # ========================================================

    brain_negative = run(
        initial_roll_rate=-INITIAL_ROLL_RATE,
        use_brain=True,
        seed=43,
    )


    print()
    print("=" * 60)
    print("VERDICT")
    print("=" * 60)

    print(
        "Baseline +1.5 :",
        f"{baseline:+.4f}"
    )

    print(
        "Brain +1.5    :",
        f"{brain_positive:+.4f}"
    )

    print(
        "Brain -1.5    :",
        f"{brain_negative:+.4f}"
    )