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


def get_d(u, N, h, dt, prices, vols, corr):
    Dxy = (u[2:N[0] + 2, 2:N[1] + 2, 1:N[2] + 1] - u[:N[0], 2:N[1] + 2, 1:N[2] + 1] - u[2:N[0] + 2, :N[1], 1:N[2] + 1] +
           u[:N[0], :N[1], 1:N[2] + 1]) / (4 * h[0] * h[1])
    Dyz = (u[1:N[0] + 1, 2:N[1] + 2, 2:N[2] + 2] - u[1:N[0] + 1, :N[1], 2:N[2] + 2] - u[1:N[0] + 1, 2:N[1] + 2, :N[2]] +
           u[1:N[0] + 1, :N[1], :N[2]]) / (4 * h[1] * h[2])
    Dzx = (u[2:N[0] + 2, 1:N[1] + 1, 2:N[2] + 2] - u[:N[0], 1:N[1] + 1, 2:N[2] + 2] - u[2:N[0] + 2, 1:N[1] + 1, :N[2]] +
           u[:N[0], 1:N[1] + 1, :N[2]]) / (4 * h[0] * h[2])
    xy = prices[0][1:N[0] + 1].reshape(N[0], 1) * prices[1][1:N[1] + 1].reshape(1, N[1])
    yz = prices[1][1:N[1] + 1].reshape(N[1], 1) * prices[2][1:N[2] + 1].reshape(1, N[2])
    zx = prices[2][1:N[2] + 1].reshape(N[2], 1) * prices[0][1:N[0] + 1].reshape(1, N[0])
    return ((vols[0] * vols[1] * corr[0][1] * xy.reshape(N[0], N[1], 1) * Dxy + vols[1] * vols[2] * corr[1][2] *
             yz.reshape(1, N[1], N[2]) * Dyz + vols[2] * vols[0] * corr[2][0] * zx.reshape(N[0], 1, N[2]) * Dzx) / 3 - 1
            / dt) * u[1:N[0] + 1, 1:N[1] + 1, 1:N[2] + 1]


def TDMAsolver(aa, bb, cc, dd, direction):
    dd = deepcopy(dd)
    mat = np.diag(bb) + np.diag(aa, k=-1) + np.diag(cc, k=1)
    mat = np.linalg.inv(mat)
    mat = np.repeat(mat[:, :, np.newaxis], len(dd), axis=2)     # dd[0]??
    mat = np.repeat(mat[:, :, :, np.newaxis], len(dd), axis=3)
    if direction == 1:
        dd = np.swapaxes(mat, 0, 1)
    if direction == 2:
        dd = np.swapaxes(mat, 0, 2)
    u = np.einsum('ijkl, ijk -> ijk', mat, dd)
    return u


def boundary_cond(u, N, redemption_dates, coupon, F, K, initial_underlyings, _date, prices):
    # 6 faces
    u[0, 1:N[1] + 1, 1:N[2] + 1] = 2 * u[1, 1:N[1] + 1, 1:N[2] + 1] - u[2, 1:N[1] + 1, 1:N[2] + 1]
    u[1:N[0] + 1, 0, 1:N[2] + 1] = 2 * u[1:N[0] + 1, 1, 1:N[2] + 1] - u[1:N[0] + 1, 2, 1:N[2] + 1]
    u[1:N[0] + 1, 1:N[1] + 1, 0] = 2 * u[1:N[0] + 1, 1:N[1] + 1, 1] - u[1:N[0] + 1, 1:N[1] + 1, 2]
    u[N[0] + 2, 1:N[1] + 1, 1:N[2] + 1] = 2 * u[N[0] + 1, 1:N[1] + 1, 1:N[2] + 1] - u[N[0], 1:N[1] + 1, 1:N[2] + 1]
    u[1:N[0] + 1, N[1] + 2, 1:N[2] + 1] = 2 * u[1:N[0] + 1, N[1] + 1, 1:N[2] + 1] - u[1:N[0] + 1, N[1], 1:N[2] + 1]
    u[1:N[0] + 1, 1:N[1] + 1, N[2] + 2] = 2 * u[1:N[0] + 1, 1:N[1] + 1, N[2] + 1] - u[1:N[0] + 1, 1:N[1] + 1, N[2]]
    # 12 lines
    u[1:N[0] + 1, 0, 0], u[0, 1:N[1] + 1, 0], u[0, 0, 1:N[2] + 1], u[N[0] + 2, 1:N[1] + 1:0] = 0, 0, 0, 0
    u[N[0] + 2, 0, 1:N[2] + 1], u[1:N[0] + 1, N[1] + 2, 0], u[0, N[1] + 2, 1:N[2] + 1] = 0, 0, 0
    u[1:N[0] + 1, 0, N[2] + 2], u[0, 1:N[1] + 1, N[2] + 2] = 0, 0
    u[1:N[0] + 1, N[1] + 2, N[2] + 2] = list(
        map(lambda x: (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F
        if x >= K[_date] * initial_underlyings[0] else 0, prices[0, 1:N[0]]))
    u[N[0] + 2, 1:N[1] + 1, N[2] + 2] = list(
        map(lambda y: (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F
        if y >= K[_date] * initial_underlyings[1] else 0, prices[1, 1:N[1]]))
    u[N[0] + 2, N[1] + 2, 1:N[2] + 1] = list(
        map(lambda z: (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F
        if z >= K[_date] * initial_underlyings[2] else 0, prices[2, 1:N[2]]))
    # 8 edges
    u[0, 0, 0], u[N[0] + 2, 0, 0], u[0, N[1] + 2, 0], u[0, 0, N[2] + 2], u[N[0] + 2, N[1] + 2, 0], u[
        0, N[1] + 2, N[2] + 2], u[N[0] + 2, 0, N[2] + 2] = 0, 0, 0, 0, 0, 0, 0
    u[N[0] + 1, N[1] + 1, N[2] + 1] = (1 + coupon[(redemption_dates[redemption_dates >= _date]).min()]) * F
    return u


def get_payoff(u, redemption_dates, coupon, F, K, initial_underlyings, _date, a, b, c, h, dt, prices, vols, corr, N):
    # x - direction
    d = get_d(u=u, N=N, h=h, dt=dt, prices=prices, vols=vols, corr=corr)
    u[1:-1, 1:-1, 1:-1] = TDMAsolver(a[0][1:], b[0], c[0][:-1], d, direction=0)

    # Linear boundary condition
    u = boundary_cond(u=u, N=N, redemption_dates=redemption_dates, coupon=coupon, F=F, K=K,
                      initial_underlyings=initial_underlyings, _date=_date, prices=prices)

    # y - direction
    u[1:-1, 1:-1, 1:-1] = TDMAsolver(a[0][1:], b[0], c[0][:-1], d, direction=1)

    # Linear boundary condition
    u = boundary_cond(u=u, N=N, redemption_dates=redemption_dates, coupon=coupon, F=F, K=K,
                      initial_underlyings=initial_underlyings, _date=_date, prices=prices)

    # z - direction
    u[1:-1, 1:-1, 1:-1] = TDMAsolver(a[0][1:], b[0], c[0][:-1], d, direction=2)

    # Linear boundary condition
    u = boundary_cond(u=u, N=N, redemption_dates=redemption_dates, coupon=coupon, F=F, K=K,
                      initial_underlyings=initial_underlyings, _date=_date, prices=prices)
    return u
