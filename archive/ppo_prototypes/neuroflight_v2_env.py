import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from archive.ppo_prototypes.neuroflight_env import NeuroFlightEnv


class NeuroFlightV2Env(NeuroFlightEnv):

    def __init__(
        self,
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    ):

        super().__init__(
            target_z=target_z,
            frame_skip=frame_skip,
            max_steps=max_steps,
        )

        # 12 signaux biologiques actuels
        # +
        # 12 variations temporelles
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(24,),
            dtype=np.float32,
        )

        self.previous_bio = np.zeros(
            12,
            dtype=np.float32,
        )

        self.previous_action = np.zeros(
            4,
            dtype=np.float32,
        )

        self.v2_step = 0


    # ========================================================
    # QUATERNION -> ROLL / PITCH / YAW
    # ========================================================

    def _rpy(self, quat):

        w, x, y, z = quat

        roll = math.atan2(
            2.0 * (w*x + y*z),
            1.0 - 2.0 * (x*x + y*y),
        )

        value = 2.0 * (w*y - z*x)

        value = np.clip(
            value,
            -1.0,
            1.0,
        )

        pitch = math.asin(value)

        yaw = math.atan2(
            2.0 * (w*z + x*y),
            1.0 - 2.0 * (y*y + z*z),
        )

        return roll, pitch, yaw


    # ========================================================
    # EULER -> QUATERNION
    # ========================================================

    def _euler_to_quat(
        self,
        roll,
        pitch,
        yaw,
    ):

        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)

        w = (
            cr*cp*cy
            + sr*sp*sy
        )

        x = (
            sr*cp*cy
            - cr*sp*sy
        )

        y = (
            cr*sp*cy
            + sr*cp*sy
        )

        z = (
            cr*cp*sy
            - sr*sp*cy
        )

        return np.array(
            [w, x, y, z],
            dtype=np.float64,
        )


    # ========================================================
    # VITESSE MONDE -> REPERE DU DRONE
    # ========================================================

    def _world_to_body(
        self,
        vector,
        quat,
    ):

        w, x, y, z = quat

        # Rotation body -> world
        R = np.array(
            [
                [
                    1 - 2*(y*y + z*z),
                    2*(x*y - z*w),
                    2*(x*z + y*w),
                ],
                [
                    2*(x*y + z*w),
                    1 - 2*(x*x + z*z),
                    2*(y*z - x*w),
                ],
                [
                    2*(x*z - y*w),
                    2*(y*z + x*w),
                    1 - 2*(x*x + y*y),
                ],
            ],
            dtype=np.float64,
        )

        # world -> body
        return R.T @ vector


    # ========================================================
    # CAPTEURS BIOLOGIQUES
    # ========================================================

    def _bio_features(self):

        gyro = np.asarray(
            self._sensor(
                self.gyro_id
            ),
            dtype=np.float32,
        )

        acceleration = np.asarray(
            self._sensor(
                self.acc_id
            ),
            dtype=np.float32,
        )

        quat = np.asarray(
            self._sensor(
                self.quat_id
            ),
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # OPTIC FLOW APPROXIMATIF
        #
        # On ne donne PAS X et Y au cerveau.
        #
        # On transforme la vitesse en mouvement visuel
        # relatif, comme ce que pourrait observer une mouche.
        # ----------------------------------------------------

        world_velocity = np.asarray(
            self.data.qvel[0:3],
            dtype=np.float64,
        )

        body_velocity = (
            self._world_to_body(
                world_velocity,
                quat,
            )
        )

        altitude = max(
            float(
                self.data.qpos[2]
            ),
            0.15,
        )


        translational_flow = (
            -body_velocity
            / altitude
        )

        translational_flow = np.clip(
            translational_flow,
            -5.0,
            5.0,
        ).astype(np.float32)


        # Vision de rotation :
        # les yeux voient aussi une rotation du monde.
        rotational_flow = np.clip(
            gyro,
            -10.0,
            10.0,
        )


        # ----------------------------------------------------
        # 12 VALEURS
        #
        # 0:3   gyro
        # 3:6   acceleration
        # 6:9   optic flow translation
        # 9:12  optic flow rotation
        # ----------------------------------------------------

        features = np.concatenate(
            [
                np.clip(
                    gyro,
                    -10.0,
                    10.0,
                ),

                np.clip(
                    acceleration,
                    -20.0,
                    20.0,
                ),

                translational_flow,

                rotational_flow,
            ]
        )

        return features.astype(
            np.float32
        )


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_obs(self):

        current = self._bio_features()

        delta = (
            current
            - self.previous_bio
        )

        delta = np.clip(
            delta,
            -10.0,
            10.0,
        )

        observation = np.concatenate(
            [
                current,
                delta,
            ]
        ).astype(np.float32)

        self.previous_bio = (
            current.copy()
        )

        return observation


    # ========================================================
    # ACTION
    # ========================================================

    def _apply_action(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(4)

        action = np.clip(
            action,
            -1.0,
            1.0,
        )


        # Poussée autour du hover
        thrust = (
            self.hover_thrust
            + float(action[0]) * 0.045
        )

        thrust = float(
            np.clip(
                thrust,
                0.0,
                0.35,
            )
        )


        # Moments plus doux que V1
        moment_x = (
            0.25
            * float(action[1])
        )

        moment_y = (
            0.25
            * float(action[2])
        )

        moment_z = (
            0.15
            * float(action[3])
        )


        self.data.ctrl[0] = thrust
        self.data.ctrl[1] = moment_x
        self.data.ctrl[2] = moment_y
        self.data.ctrl[3] = moment_z


    # ========================================================
    # REWARD V2
    # ========================================================

    def _reward_v2(
        self,
        action,
    ):

        quat = np.asarray(
            self._sensor(
                self.quat_id
            ),
            dtype=np.float64,
        )

        roll, pitch, _ = (
            self._rpy(quat)
        )

        gyro = np.asarray(
            self._sensor(
                self.gyro_id
            ),
            dtype=np.float64,
        )

        velocity = np.asarray(
            self.data.qvel[0:3],
            dtype=np.float64,
        )

        z = float(
            self.data.qpos[2]
        )


        # ----------------------------------------------------
        # OBJECTIF PRINCIPAL :
        # SURVIVRE EN VOL
        # ----------------------------------------------------

        reward = 1.0


        # Rester relativement horizontal
        tilt_error = (
            roll**2
            + pitch**2
        )

        reward -= (
            3.0
            * tilt_error
        )


        # Eviter les rotations violentes
        reward -= (
            0.08
            * float(
                np.sum(
                    gyro**2
                )
            )
        )


        # Eviter de partir trop vite
        horizontal_speed = (
            velocity[0]**2
            + velocity[1]**2
        )

        reward -= (
            0.10
            * horizontal_speed
        )


        # Limiter montée/chute brutale
        reward -= (
            0.20
            * velocity[2]**2
        )


        # ----------------------------------------------------
        # ALTITUDE
        #
        # On ne lui impose PLUS exactement 0.50 m.
        #
        # On veut principalement qu'il reste en l'air.
        # ----------------------------------------------------

        if z < 0.20:

            reward -= (
                4.0
                * (0.20 - z)**2
            )

        if z > 1.00:

            reward -= (
                1.5
                * (z - 1.00)**2
            )


        # ----------------------------------------------------
        # COMMANDES TROP FORTES
        # ----------------------------------------------------

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        reward -= (
            0.003
            * float(
                np.sum(
                    action**2
                )
            )
        )


        # ----------------------------------------------------
        # EVITER LES CHANGEMENTS BRUTAUX
        # ----------------------------------------------------

        action_change = (
            action
            - self.previous_action
        )

        reward -= (
            0.01
            * float(
                np.sum(
                    action_change**2
                )
            )
        )

        self.previous_action = (
            action.copy()
        )

        return float(reward)


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        gym.Env.reset(
            self,
            seed=seed,
        )

        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        # Position
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.50


        # Perturbation légère initiale
        roll = math.radians(
            self.np_random.uniform(
                -8.0,
                8.0,
            )
        )

        pitch = math.radians(
            self.np_random.uniform(
                -8.0,
                8.0,
            )
        )

        yaw = math.radians(
            self.np_random.uniform(
                -10.0,
                10.0,
            )
        )


        self.data.qpos[3:7] = (
            self._euler_to_quat(
                roll,
                pitch,
                yaw,
            )
        )


        # Petite vitesse initiale
        self.data.qvel[:] = 0.0

        self.data.qvel[0] = (
            self.np_random.uniform(
                -0.05,
                0.05,
            )
        )

        self.data.qvel[1] = (
            self.np_random.uniform(
                -0.05,
                0.05,
            )
        )

        self.data.qvel[2] = (
            self.np_random.uniform(
                -0.03,
                0.03,
            )
        )


        mujoco.mj_forward(
            self.model,
            self.data,
        )


        current = (
            self._bio_features()
        )

        self.previous_bio = (
            current.copy()
        )

        self.previous_action[:] = 0.0

        self.v2_step = 0


        observation = np.concatenate(
            [
                current,
                np.zeros_like(
                    current
                ),
            ]
        ).astype(np.float32)


        return observation, {}


    # ========================================================
    # STEP
    # ========================================================

    def step(
        self,
        action,
    ):

        self._apply_action(
            action
        )


        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )


        self.v2_step += 1


        observation = (
            self._get_obs()
        )

        reward = (
            self._reward_v2(
                action
            )
        )


        x = float(
            self.data.qpos[0]
        )

        y = float(
            self.data.qpos[1]
        )

        z = float(
            self.data.qpos[2]
        )


        quat = np.asarray(
            self._sensor(
                self.quat_id
            )
        )

        roll, pitch, yaw = (
            self._rpy(quat)
        )


        distance = math.sqrt(
            x*x
            + y*y
        )


        terminated = bool(
            z < 0.05
            or
            z > 1.50
            or
            distance > 3.0
            or
            abs(roll)
            > math.radians(80)
            or
            abs(pitch)
            > math.radians(80)
        )


        truncated = bool(
            self.v2_step
            >= self.max_steps
        )


        if terminated:
            reward -= 20.0


        # Grosse récompense :
        # il a tenu tout l'épisode.
        if (
            truncated
            and not terminated
        ):
            reward += 25.0


        info = {
            "x": x,
            "y": y,
            "z": z,

            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,

            "survived_steps":
                self.v2_step,
        }


        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )