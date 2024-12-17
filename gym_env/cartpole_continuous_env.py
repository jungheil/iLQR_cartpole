# coding: utf-8
# Copyright (c) 2024 Jungheil <jungheilai@gmail.com>
# iLQR_cartpole is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.

import numpy as np
from gym import logger, spaces
from gym.envs.classic_control import CartPoleEnv


class CartPoleContinuousEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()

        # direction & scale of force magnitude.
        self.force_mag = 15.0
        self.min_action = np.float32(-1.0)
        self.max_action = np.float32(1.0)
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        return_info=False,
        options=None,
    ):
        super().reset(seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state[2] = self.np_random.uniform(low=-np.pi, high=np.pi)
        # self.state[2] = np.pi / 2
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        # Continuous Case:
        force = self.force_mag * action[0]

        # Note: everything below this is same as gym's cartpole step fun.
        x, x_dot, theta, theta_dot = tuple(self.state)
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float32)

        done = (
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True."
                    " You should always call 'reset()' once you receive 'done = True' -- any further steps "
                    "are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done, {}
