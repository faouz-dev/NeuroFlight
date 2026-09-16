import math

import gymnasium as gym
import mujoco
import mujoco_menagerie as mm
import numpy as np


class NeuroFlightEnv(gym.Env):

    metadata = {
        "render_modes": ["human"],
        "render_fps": 100,
    }

    def __init__(
        self,
        render_mode=None,
        target_z=0.50,
        frame_skip=5,
        max_steps=1000,
    ):

        super().__init__()

        # ====================================================
        # MUJOCO
        # ====================================================

        self.model = mm.load(
            "bitcraze_crazyflie_2"
        )

        self.data = mujoco.MjData(
            self.model
        )

        self.render_mode = render_mode
        self.viewer = None

        self.frame_skip = frame_skip
        self.max_steps = max_steps

        self.target_z = target_z

        self.mass = float(
            self.model.body_mass.sum()
        )

        self.gravity = abs(
            float(self.model.opt.gravity[2])
        )

        self.hover_thrust = (
            self.mass * self.gravity
        )

        self.step_count = 0


        # ====================================================
        # ACTIONS
        #
        # Réseau neuronal :
        #
        # action[0] = thrust
        # action[1] = roll moment
        # action[2] = pitch moment
        # action[3] = yaw moment
        #
        # Toutes les sorties sont normalisées entre -1 et 1.
        # ====================================================

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )


        # ====================================================
        # OBSERVATIONS
        #
        # 16 valeurs :
        #
        # position       3
        # vitesse       3
        # quaternion    4
        # gyroscope     3
        # accélération  3
        #
        # TOTAL = 16
        # ====================================================

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32,
        )


        # ====================================================
        # CAPTEURS
        # ====================================================

        self.gyro_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "body_gyro",
        )

        self.acc_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "body_linacc",
        )

        self.quat_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            "body_quat",
        )


    # ========================================================
    # CAPTEURS
    # ========================================================

    def _sensor(self, sensor_id):

        address = self.model.sensor_adr[
            sensor_id
        ]

        dimension = self.model.sensor_dim[
            sensor_id
        ]

        return self.data.sensordata[
            address:
            address + dimension
        ].copy()


    # ========================================================
    # OBSERVATION
    # ========================================================

    def _get_observation(self):

        position = np.array([
            self.data.qpos[0],
            self.data.qpos[1],

            # erreur d'altitude plutôt
            # que hauteur absolue
            self.data.qpos[2]
            - self.target_z,
        ])

        velocity = np.array([
            self.data.qvel[0],
            self.data.qvel[1],
            self.data.qvel[2],
        ])

        quat = self._sensor(
            self.quat_id
        )

        gyro = self._sensor(
            self.gyro_id
        )

        acceleration = self._sensor(
            self.acc_id
        )

        observation = np.concatenate([
            position,
            velocity,
            quat,
            gyro,
            acceleration,
        ])

        return observation.astype(
            np.float32
        )


    # ========================================================
    # QUATERNION -> ROLL/PITCH
    # ========================================================

    def _orientation(self):

        quat = self._sensor(
            self.quat_id
        )

        w, x, y, z = quat

        roll = math.atan2(
            2 * (w*x + y*z),
            1 - 2 * (x*x + y*y),
        )

        pitch_value = (
            2 * (w*y - z*x)
        )

        pitch_value = np.clip(
            pitch_value,
            -1.0,
            1.0,
        )

        pitch = math.asin(
            pitch_value
        )

        return roll, pitch


    # ========================================================
    # ACTION IA -> ACTION MUJOCO
    # ========================================================

    def _apply_action(self, action):

        # Conversion propre en tableau 1D
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(4)

        # PPO travaille dans [-1, +1]
        action = np.clip(
            action,
            -1.0,
            1.0,
        )


        # ========================================================
        # THRUST
        #
        # action[0] = 0
        # -> poussée nécessaire au hover
        #
        # action[0] = -1
        # -> diminution de poussée
        #
        # action[0] = +1
        # -> augmentation de poussée
        # ========================================================

        thrust = (
            self.hover_thrust
            + float(action[0]) * 0.05
        )

        thrust = float(
            np.clip(
                thrust,
                0.0,
                0.35,
            )
        )


        # ========================================================
        # MOMENTS
        #
        # On réduit volontairement leur amplitude pour éviter
        # que l'IA fasse immédiatement des corrections énormes.
        # ========================================================

        moment_x = (
            0.35 * float(action[1])
        )

        moment_y = (
            0.35 * float(action[2])
        )

        moment_z = (
            0.25 * float(action[3])
        )


        # ========================================================
        # COMMANDES MUJOCO
        # ========================================================

        self.data.ctrl[0] = thrust
        self.data.ctrl[1] = moment_x
        self.data.ctrl[2] = moment_y
        self.data.ctrl[3] = moment_z


    # ========================================================
    # REWARD
    # ========================================================

    def _reward(self, action):

        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        z = float(self.data.qpos[2])

        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        vz = float(self.data.qvel[2])

        roll, pitch = (
            self._orientation()
        )

        gyro = self._sensor(
            self.gyro_id
        )


        # Erreurs
        altitude_error = (
            z - self.target_z
        )

        horizontal_distance = (
            x*x + y*y
        )

        linear_speed = (
            vx*vx
            + vy*vy
            + vz*vz
        )

        angular_speed = float(
            np.sum(gyro * gyro)
        )

        tilt = (
            roll*roll
            + pitch*pitch
        )

        control_effort = float(
            np.sum(
                np.asarray(action) ** 2
            )
        )


        # ====================================================
        # SCORE
        # ====================================================

        reward = 1.0

        reward -= (
            4.0 * tilt
        )

        reward -= (
            3.0
            * altitude_error**2
        )

        reward -= (
            0.40
            * horizontal_distance
        )

        reward -= (
            0.10
            * linear_speed
        )

        reward -= (
            0.03
            * angular_speed
        )

        reward -= (
            0.002
            * control_effort
        )

        return float(reward)


    # ========================================================
    # CRASH / FIN EPISODE
    # ========================================================

    def _terminated(self):

        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])
        z = float(self.data.qpos[2])

        roll, pitch = (
            self._orientation()
        )


        # Drone au sol
        if z < 0.05:
            return True


        # Drone beaucoup trop haut
        if z > 2.0:
            return True


        # Parti trop loin
        distance = math.sqrt(
            x*x + y*y
        )

        if distance > 3.0:
            return True


        # Presque retourné
        max_angle = math.radians(
            80
        )

        if (
            abs(roll) > max_angle
            or abs(pitch) > max_angle
        ):
            return True


        return False


    # ========================================================
    # RESET
    # ========================================================

    def reset(
        self,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        mujoco.mj_resetData(
            self.model,
            self.data,
        )


        # Position initiale
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.target_z


        # ====================================================
        # PERTURBATION ALEATOIRE
        #
        # L'IA ne verra jamais exactement
        # le même départ.
        # ====================================================

        roll = self.np_random.uniform(
            math.radians(-10),
            math.radians(10),
        )

        pitch = self.np_random.uniform(
            math.radians(-10),
            math.radians(10),
        )

        yaw = self.np_random.uniform(
            math.radians(-10),
            math.radians(10),
        )


        # Euler -> quaternion
        cr = math.cos(roll / 2)
        sr = math.sin(roll / 2)

        cp = math.cos(pitch / 2)
        sp = math.sin(pitch / 2)

        cy = math.cos(yaw / 2)
        sy = math.sin(yaw / 2)


        self.data.qpos[3] = (
            cr*cp*cy + sr*sp*sy
        )

        self.data.qpos[4] = (
            sr*cp*cy - cr*sp*sy
        )

        self.data.qpos[5] = (
            cr*sp*cy + sr*cp*sy
        )

        self.data.qpos[6] = (
            cr*cp*sy - sr*sp*cy
        )


        # Petite vitesse initiale aléatoire
        self.data.qvel[0] = (
            self.np_random.uniform(
                -0.15,
                0.15,
            )
        )

        self.data.qvel[1] = (
            self.np_random.uniform(
                -0.15,
                0.15,
            )
        )


        mujoco.mj_forward(
            self.model,
            self.data,
        )

        self.step_count = 0


        observation = (
            self._get_observation()
        )


        return observation, {}


    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        self._apply_action(
            action
        )


        # Une action reste active pendant
        # plusieurs pas physiques.
        for _ in range(
            self.frame_skip
        ):

            mujoco.mj_step(
                self.model,
                self.data,
            )


        self.step_count += 1


        observation = (
            self._get_observation()
        )

        reward = self._reward(
            action
        )

        terminated = (
            self._terminated()
        )

        truncated = (
            self.step_count
            >= self.max_steps
        )


        if terminated:
            reward -= 20.0


        return (
            observation,
            reward,
            terminated,
            truncated,
            {},
        )


    # ========================================================
    # FERMETURE
    # ========================================================

    def close(self):

        if self.viewer is not None:
            self.viewer.close()

            self.viewer = None