# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 
"""
import numpy as np
from simulation_method.FDM.IFDM import TDMAsolver

T = 0.5
Nt = 1600
dt = T / Nt
x0 = 10
Nx = 160
dx = x0 / Nx
price = np.linspace(0, 2 * x0, Nx + 2)
r = 0.05
q = 0
sig = 0.2
k = 10

u = np.zeros((Nt, Nx + 2))

# coefficient
a = dt * (r + sig ** 2 * price ** 2) / 2
b = dt * ((r - q) * price - sig ** 2 * price ** 2) / 4
c = -dt * ((r - q) * price + sig ** 2 * price ** 2) / 4
d = np.zeros(Nx + 1)

# initial setting
for x in range(Nx + 2):
    u[0, x] = max(price[x] - k, 0)

for t in range(Nt):
    u[t, 0] = max(price[0] - k, 0)
    u[t, -1] = max(price[-1] - k, 0)

# solve eq
for t in range(Nt):
    for x in range(1, Nx + 1):
        d[x] = -b[x] * u[t, x - 1] + (1 - a[x]) * u[t, x] - c[x] * u[t, x + 1]
    if t < Nt - 1:
        u[t + 1, :-1] = TDMAsolver(b[2:], 1 + a[1:], c[1:-1], d)
