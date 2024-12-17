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

import datetime
import os
from functools import partial

import gym
import jax
import jax.numpy as jnp
from PIL import Image

import gym_env
from ilqr import ILQR

now = datetime.datetime.now()
timenow = f'{now:%Y_%m_%d_%H_%M_%S}'

@jax.jit
def dynamics(state, action):
    force_mag = 15.0
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0

    force = force_mag * action[0]

    x, x_dot, theta, theta_dot = state
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    theta_acc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * costheta / total_mass

    return jnp.array((x_dot, x_acc, theta_dot, theta_acc))


class CartPoleILQR(ILQR):
    @staticmethod
    def _theta_range(x):
        def _add_2pi(x):
            return x + 2 * jnp.pi

        def _sub_2pi(x):
            return x - 2 * jnp.pi

        jax.lax.cond(x > jnp.pi, _sub_2pi, lambda x: x, operand=x)
        jax.lax.cond(x < -jnp.pi, _add_2pi, lambda x: x, operand=x)
        return x

    @partial(jax.jit, static_argnums=0)
    def stage_cost(self, x, u, x_ref, u_ref):
        x = x.at[2].apply(CartPoleILQR._theta_range)
        return 0.5 * jnp.matmul(
            (x - x_ref).T, jnp.matmul(self.Q, (x - x_ref))
        ) + 0.5 * jnp.matmul((u - u_ref).T, jnp.matmul(self.R, (u - u_ref)))

    @partial(jax.jit, static_argnums=0)
    def term_cost(self, x, x_ref):
        x = x.at[2].apply(CartPoleILQR._theta_range)
        return 0.5 * jnp.matmul((x - x_ref).T, jnp.matmul(self.Qf, (x - x_ref)))


save_log = False
debug = True
total_iter = 500

ilqr_iter = 8
N = 30
dt = 0.02
Q = jnp.array(
    [
        [0.01, 0.0, 0.0, 0.0],
        [0.0, 0.0001, 0.0, 0.0],
        [0.0, 0.0, 0.01, 0.0],
        [0.0, 0.0, 0.0, 0.0001],
    ],
    dtype=jnp.float32,
)
R = jnp.array([[0.05]], dtype=jnp.float32)
Qf = jnp.array(
    [
        [1, 0.0, 0.0, 0.0],
        [0.0, 1.5, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=jnp.float32,
)

env = gym.make("CartPoleContinuous-v1").env
obs = env.reset()

ilqr = CartPoleILQR(dynamics, Q, R, Qf, N, dt, (jnp.array((-1,)), jnp.array((1,))))

X_ref = [jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32) for _ in range(N)]
U_ref = [jnp.array([0.0], dtype=jnp.float32) for _ in range(N - 1)]
x0 = jnp.array(obs)
U = U_ref.copy()


if save_log:
    os.makedirs(f'log/{timenow}', exist_ok=True)
    os.makedirs(f'log/{timenow}/image', exist_ok=True)
    open(f"log/{timenow}/data.csv", "w")

print('compiling... please wait a second.')
for cnt in range(total_iter):
    env.render()

    X, U = ilqr(x0, U, X_ref, U_ref, max_iter=ilqr_iter)

    if debug:
        print(f'cnt: {cnt}:')
        print(f'x: {X[0]}, u: {U[0]}')
        print(f'cost: {ilqr.trajectory_cost(X, U, X_ref, U_ref)}')
    if save_log:
        img = env.render(mode="rgb_array")
        img = Image.fromarray(img)
        img.save(f"log/{timenow}/image/frame_%04d.png" % cnt)

        with open(f"log/{timenow}/data.csv", "a") as f:
            f.write(f'{cnt},{X[0][0]},{X[0][1]},{X[0][2]},{X[0][3]},{U[0][0]}\n')

    obs, _, _, _ = env.step(U[0])

    x0 = jnp.array(obs)
    U[:-1] = U[1:]
    U[-1] = jnp.array((0,), dtype=jnp.float32)
