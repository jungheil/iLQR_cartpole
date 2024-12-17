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

from functools import partial

import jax
import jax.numpy as jnp


def rk4(dt, index=0):
    def _rk4(func):
        def wrapper(*args):
            k1 = dt * func(*args)
            args2 = tuple(
                [a if i != index else a + k1 * 0.5 for i, a in enumerate(args)]
            )
            k2 = dt * func(*args2)
            args3 = tuple(
                [a if i != index else a + k2 * 0.5 for i, a in enumerate(args)]
            )
            k3 = dt * func(*args3)
            args4 = tuple([a if i != index else a + k3 for i, a in enumerate(args)])
            k4 = dt * func(*args4)
            return args[index] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

        return wrapper

    return _rk4


class ILQR:
    def __init__(self, dynamics, Q, R, Qf, N, dt, u_space) -> None:
        self.dynamics = dynamics
        self.dynamics_discrete = rk4(dt)(dynamics)

        self.Q = Q
        self.R = R
        self.Qf = Qf

        self.N = N
        self.u_space = u_space

        self.A_jac = jax.jit(jax.jacrev(self.dynamics_discrete, argnums=0))
        self.B_jac = jax.jit(jax.jacrev(self.dynamics_discrete, argnums=1))

    def dynamics_jacobians(self, x, u):
        return self.A_jac(x, u), self.B_jac(x, u)

    @partial(jax.jit, static_argnums=0)
    def stage_cost(self, x, u, x_ref, u_ref):
        return 0.5 * jnp.matmul(
            (x - x_ref).T, jnp.matmul(self.Q, (x - x_ref))
        ) + 0.5 * jnp.matmul((u - u_ref).T, jnp.matmul(self.R, (u - u_ref)))

    @partial(jax.jit, static_argnums=0)
    def term_cost(self, x, x_ref):
        return 0.5 * jnp.matmul((x - x_ref).T, jnp.matmul(self.Qf, (x - x_ref)))

    @partial(jax.jit, static_argnums=0)
    def trajectory_cost(self, X, U, X_ref, U_ref):
        J = 0.0
        for i in range(self.N - 1):
            J += self.stage_cost(X[i], U[i], X_ref[i], U_ref[i])
        J += self.term_cost(X[-1], X_ref[-1])
        return J

    @partial(jax.jit, static_argnums=0)
    def backward_pass(self, X, U, X_ref, U_ref):
        p = [jnp.zeros(len(X[0])) for _ in range(self.N)]
        P = [jnp.zeros((len(X[0]), len(X[0]))) for _ in range(self.N)]
        d = [jnp.zeros(len(U[0])) for _ in range(self.N - 1)]
        K = [jnp.zeros((len(U[0]), len(X[0]))) for _ in range(self.N - 1)]
        p[-1], P[-1] = jnp.matmul(self.Qf, (X[-1] - X_ref[-1])), self.Qf

        dJ = 0
        for i in range(self.N - 1)[::-1]:
            lx = jnp.matmul(self.Q, (X[i] - X_ref[i]))
            lxx = self.Q
            lu = jnp.matmul(self.R, (U[i] - U_ref[i]))
            luu = self.R

            A, B = self.dynamics_jacobians(X[i], U[i])

            gx = lx + jnp.matmul(A.T, p[i + 1])
            gu = lu + jnp.matmul(B.T, p[i + 1])
            Gxx = lxx + jnp.matmul(A.T, jnp.matmul(P[i + 1], A))
            Guu = luu + jnp.matmul(B.T, jnp.matmul(P[i + 1], B))
            Gxu = jnp.matmul(A.T, jnp.matmul(P[i + 1], B))
            Gux = jnp.matmul(B.T, jnp.matmul(P[i + 1], A))

            Guu_i = jnp.linalg.inv(Guu)
            d[i] = jnp.matmul(Guu_i, gu)
            K[i] = jnp.matmul(Guu_i, Gux)

            p[i] = (
                gx
                - jnp.matmul(K[i].T, gu)
                + jnp.matmul(K[i].T, jnp.matmul(Guu, d[i]))
                - jnp.matmul(Gxu, d[i])
            )
            P[i] = (
                Gxx
                + jnp.matmul(K[i].T, jnp.matmul(Guu, K[i]))
                - jnp.matmul(K[i].T, Gux)
                - jnp.matmul(Gxu, K[i])
            )

            dJ += jnp.matmul(gu.T, d[i])

        return d, K, dJ

    @partial(jax.jit, static_argnums=0)
    def forward_pass(self, X, U, X_ref, U_ref, d, K, dJ, max_iter=8):
        J = self.trajectory_cost(X, U, X_ref, U_ref)

        def _loop_body(st):
            iter, alpha, Jn, Xn, Un = st
            Jn = 0.0
            for i in range(self.N - 1):
                Un[i] = U[i] - alpha * d[i] - jnp.matmul(K[i], (Xn[i] - X[i]))
                Un[i] = jnp.clip(Un[i], self.u_space[0], self.u_space[1])
                Xn[i + 1] = self.dynamics_discrete(Xn[i], Un[i])
                Jn += self.stage_cost(Xn[i], Un[i], X_ref[i], U_ref[i])
            Jn += self.term_cost(Xn[-1], X_ref[-1])
            alpha /= 2.0
            iter += 1
            return (iter, alpha, Jn, Xn, Un)

        def _loop_cond(st):
            iter, alpha, Jn, Xn, Un = st
            return jnp.logical_and(
                iter < max_iter, jnp.logical_or(jnp.isnan(Jn), Jn > J - alpha * dJ)
            )

        iter, alpha, Jn, Xn, Un = jax.lax.while_loop(
            _loop_cond, _loop_body, (0, 1.0, jnp.inf, X.copy(), U.copy())
        )

        return Xn, Un

    def __call__(self, x0, U, X_ref, U_ref, atol=1e-5, max_iter=100):
        x0 = jnp.array(x0)
        U = [jnp.array(u) for u in U]
        X_ref = [jnp.array(x) for x in X_ref]
        U_ref = [jnp.array(u) for u in U_ref]

        X = [x0 for _ in range(self.N)]

        for i in range(self.N - 1):
            X[i + 1] = self.dynamics_discrete(X[i], U[i])

        for _ in range(max_iter):
            d, K, dJ = self.backward_pass(X, U, X_ref, U_ref)
            Xn, Un = self.forward_pass(X, U, X_ref, U_ref, d, K, dJ)

            if jnp.linalg.norm(jnp.array(d), jnp.inf) < atol:
                break

            X = Xn.copy()
            U = Un.copy()

        return Xn, Un
