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

u = np.zeros((Nt + 1, Nx + 1))

# coefficient
a = 1 + dt * (r + sig ** 2 * price ** 2)
b = dt * ((r - q) * price - sig ** 2 * price ** 2) / 2
c = dt * (-(r - q) * price - sig ** 2 * price ** 2) / 2

# initial setting
for x in range(Nx + 1):
    u[0, x] = max(price[x] - k, 0)

for t in range(Nt + 1):
    u[t, 0] = max(price[0] - k, 0)
    u[t, Nx] = max(price[Nx] - k, 0)


# thomas algorithm
def TDMAsolver(a, b, c, d):
    nf = len(d)  # number of edivuations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy the array
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    #        del bc, cc, dc  # delete variables from memory
    return xc


# solve eq
for t in range(1, Nt + 1):
    u[t, :] = TDMAsolver(b[1:], a, c[:-1], u[t - 1, :])
