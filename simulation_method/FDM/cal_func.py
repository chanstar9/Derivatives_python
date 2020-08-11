# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 08 .10
"""

from copy import deepcopy
import numpy as np


def get_coefficient(h, dt, vols, price, r, q):
    a = -(vols * price / h) ** 2 / 2 + (r - q) * price / (2 * h)
    b = (vols * price / h) ** 2 + 1 / dt
    c = -(vols * price / h) ** 2 / 2 - (r - q) * price / (2 * h)
    return a, b, c


def TDMAsolver(aa, bb, cc, dd):
    nf = len(dd[0])  # number of edivuations
    acc, bcc, ccc, dcc = map(np.array, (aa, bb, cc, dd))  # copy the array
    for it in range(1, nf):
        mc = acc[it - 1] / bcc[it - 1]
        bcc[it] = bcc[it] - mc * ccc[it - 1]
        dcc[:, it] = dcc[:, it] - mc * dcc[:, it - 1]

    xcc = dcc
    xcc[:, -1] = dcc[:, -1] / bcc[-1]

    for il in reversed(range(0, nf - 1)):
        xcc[:, il] = (dcc[:, il] - ccc[il] * xcc[:, il + 1]) / bcc[il]

    return xcc


def ELS_initial_cond(initial_underlyings, K, redemption_dates, coupon, F, barrier_rate, prices, u, x, y, z, KI):
    if min(prices[0][x] / initial_underlyings[0], prices[1][y] / initial_underlyings[1],
           prices[2][z] / initial_underlyings[2]) >= K[redemption_dates[-1]]:
        u[x, y, z] = (1 + coupon[redemption_dates[-1]]) * F
    else:
        if not KI:  # If a knock-in does not occur before maturity
            # If a knock-in does not occur at maturity
            if min(prices[0][x] / initial_underlyings[0], prices[1][y] / initial_underlyings[1],
                   prices[2][z] / initial_underlyings[2]) >= barrier_rate:
                u[x, y, z] = (1 + coupon[redemption_dates[-1]]) * F
            else:  # If a knock-in does occur at maturity
                u[x, y, z] = min(prices[0][x] / initial_underlyings[0], prices[1][y] / initial_underlyings[1],
                                 prices[2][z] / initial_underlyings[2]) * F
        else:
            u[x, y, z] = min(prices[0][x] / initial_underlyings[0], prices[1][y] / initial_underlyings[1],
                             prices[2][z] / initial_underlyings[2]) * F
    return u


def get_d(u, x, y, z, h, dt, prices, vols, corr):
    Dxy = (u[x + 1, y + 1, z] - u[x - 1, y + 1, z] - u[x + 1, y - 1, z] + u[x - 1, y - 1, z]) / (4 * h[0] * h[1])
    Dyz = (u[x, y + 1, z + 1] - u[x, y - 1, z + 1] - u[x, y + 1, z - 1] + u[x, y - 1, z - 1]) / (4 * h[1] * h[2])
    Dzx = (u[x + 1, y, z + 1] - u[x - 1, y, z + 1] - u[x + 1, y, z - 1] + u[x - 1, y, z - 1]) / (4 * h[0] * h[2])
    return (vols[0] * vols[1] * corr[0][1] * prices[0][x] * prices[1][y] * Dxy + vols[1] * vols[2] * corr[1][2] *
            prices[1][y] * prices[2][z] * Dyz + vols[2] * vols[0] * corr[2][0] * prices[2][z] * prices[0][
                x] * Dzx) / 3 - u[x, y, z] / dt


def get_payoff(u, redemption_dates, coupon, F, _date, a, b, c, h, dt, prices, vols, corr, N):
    # x - direction
    d = np.zeros(tuple(N))
    for x in range(1, N[0] + 1):
        for y in range(1, N[1] + 1):
            for z in range(1, N[2] + 1):
                d[x - 1] = get_d(u, x, y, z, h, dt, prices, vols, corr)
    u[1:N[0] + 1, 1:-1] = TDMAsolver(a[0][1:], b[0], c[0][:-1], d)

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F

    # y - direction
    d = np.zeros((N[0], N[1]))
    for x in range(1, N[0] + 1):
        d[x - 1] = np.fromiter((get_d(u, x, y, vols, corr) for y in range(1, N[1] + 1)), dtype=float)
    u[1:-1, 1:N[1] + 1] = TDMAsolver(b[1][1:], a[1][:], c[1][:-1], d)

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F

    # z - direction

    return u
