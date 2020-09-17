# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date:
"""

import numpy as np

T = 0.5
Nt = 1600
dt = T / Nt
x0 = 10
Nx = 160
dx = x0 / Nx
price = np.linspace(0, 2 * x0, Nx + 1)
times = np.linspace(0, T, Nt + 1)
r = 0.05
q = 0
sig = 0.2
k = 10


def EFDM():
    u = np.zeros((Nt + 1, Nx + 1))

    # coefficient
    a = 1 - dt * (r + sig ** 2 * price ** 2)
    b = dt * (-(r - q) * price + sig ** 2 * price ** 2) / 2
    c = dt * ((r - q) * price + sig ** 2 * price ** 2) / 2

    # initial setting
    for x in range(Nx + 1):
        u[0, x] = max(price[x] - k, 0)

    for t in range(Nt + 1):
        u[t, 0] = max(price[0] - k, 0)
        u[t, Nx] = max(price[Nx] - k, 0)

    # solve eq
    for t in range(Nt):
        for x in range(Nx):
            u[t + 1, x] = b[x] * u[t - 1, x] + a[x] * u[t, x] + c[x] * u[t, x + 1]
    return u
