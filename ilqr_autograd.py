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

import copy

import autograd.numpy as np
from autograd import jacobian
import time

def get_time(f):

    def inner(*arg,**kwarg):
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner


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

    
    def dynamics_jacobians(self, x, u):
        A = jacobian(self.dynamics_discrete, 0)(x, u)
        B = jacobian(self.dynamics_discrete, 1)(x, u)
        return A, B
    
    def stage_cost(self, x, u, x_ref, u_ref):
        return 0.5 * np.matmul(
            (x - x_ref).T, np.matmul(self.Q, (x - x_ref))
        ) + 0.5 * np.matmul((u - u_ref).T, np.matmul(self.R, (u - u_ref)))

    def term_cost(self, x, x_ref):
        return 0.5 * np.matmul((x - x_ref).T, np.matmul(self.Qf, (x - x_ref)))

    def trajectory_cost(self, X, U, X_ref, U_ref):
        J = 0.0
        for i in range(self.N - 1):
            J += self.stage_cost(X[i], U[i], X_ref[i], U_ref[i])
        J += self.term_cost(X[-1], X_ref[-1])
        return J

    def backward_pass(self, X, U, X_ref, U_ref):
        p = [np.zeros(len(X[0])) for _ in range(self.N)]
        P = [np.zeros((len(X[0]), len(X[0]))) for _ in range(self.N)]
        d = [np.zeros(len(U[0])) for _ in range(self.N - 1)]
        K = [np.zeros((len(U[0]), len(X[0]))) for _ in range(self.N - 1)]
        p[-1], P[-1] = np.matmul(self.Qf, (X[-1] - X_ref[-1])), self.Qf

        dJ = 0
        for i in range(self.N - 1)[::-1]:
            lx = np.matmul(self.Q, (X[i] - X_ref[i]))
            lxx = self.Q
            lu = np.matmul(self.R, (U[i] - U_ref[i]))
            luu = self.R

            A, B = self.dynamics_jacobians(X[i], U[i])

            gx = lx + np.matmul(A.T, p[i + 1])
            gu = lu + np.matmul(B.T, p[i + 1])
            Gxx = lxx + np.matmul(A.T, np.matmul(P[i + 1], A))
            Guu = luu + np.matmul(B.T, np.matmul(P[i + 1], B))
            Gxu = np.matmul(A.T, np.matmul(P[i + 1], B))
            Gux = np.matmul(B.T, np.matmul(P[i + 1], A))

            Guu_i = np.linalg.inv(Guu)
            d[i] = np.matmul(Guu_i, gu)
            K[i] = np.matmul(Guu_i, Gux)

            p[i] = (
                gx
                - np.matmul(K[i].T, gu)
                + np.matmul(K[i].T, np.matmul(Guu, d[i]))
                - np.matmul(Gxu, d[i])
            )
            P[i] = (
                Gxx
                + np.matmul(K[i].T, np.matmul(Guu, K[i]))
                - np.matmul(K[i].T, Gux)
                - np.matmul(Gxu, K[i])
            )

            dJ += np.matmul(gu.T, d[i])

        return d, K, dJ

    # @get_time
    def forward_pass(self, X, U, X_ref, U_ref, d, K, dJ, max_iter=8):
        Xn = copy.deepcopy(X)
        Un = copy.deepcopy(U)
        alpha = 1.0
        J = self.trajectory_cost(X, U, X_ref, U_ref)
        Jn = 0.0
        for i in range(self.N - 1):
            Un[i] = U[i] - d[i] - np.matmul(K[i], (Xn[i] - X[i]))
            Un[i] = np.clip(Un[i], self.u_space[0], self.u_space[1], dtype=np.float32)
            Xn[i + 1] = self.dynamics_discrete(Xn[i], Un[i])
            Jn += self.stage_cost(Xn[i], Un[i], X_ref[i], U_ref[i])
        Jn += self.term_cost(Xn[-1], X_ref[-1])
        iter = 0
        while iter < max_iter and (np.isnan(Jn) or Jn > (J - 1e-2 * alpha * dJ)):
            alpha /= 2.0
            Jn = 0.0
            iter += 1
            for i in range(self.N - 1):
                Un[i] = U[i] - alpha * d[i] - np.matmul(K[i], (Xn[i] - X[i]))
                Un[i] = np.clip(
                    Un[i], self.u_space[0], self.u_space[1], dtype=np.float32
                )
                Xn[i + 1] = self.dynamics_discrete(Xn[i], Un[i])
                Jn += self.stage_cost(Xn[i], Un[i], X_ref[i], U_ref[i])
            Jn += self.term_cost(Xn[-1], X_ref[-1])
        return Xn, Un

    def __call__(self, x0, U, X_ref, U_ref, atol=1e-5, max_iter=100):
        X = [x0 for _ in range(self.N)]

        for i in range(self.N - 1):
            X[i + 1] = self.dynamics_discrete(X[i], U[i])

        for _ in range(max_iter):
            d, K, dJ = self.backward_pass(X, U, X_ref, U_ref)
            Xn, Un = self.forward_pass(X, U, X_ref, U_ref, d, K, dJ)

            if np.linalg.norm(d, np.inf) < atol:
                break

            X = copy.deepcopy(Xn)
            U = copy.deepcopy(Un)

        return Xn, Un
