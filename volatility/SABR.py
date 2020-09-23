# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 07. 28
"""

import numpy as np


# trick for using some variables in function like global variables
def SetVar(input_s, input_Tau, input_r, input_q):
    global s, tow, r, q
    s = input_s
    tow = input_Tau
    r = input_r
    q = input_q


def SABR_vol(K, alp, bet, rho, nu):
    """
    :param K:
    :param tow:
    :param 0 <= alp: control the height of the ATM implied volatility level
    :param 0 <= bet <= 1: controls curvature
    :param -1 <= rho <= 1: instantaneous correlation between underlying and its volatility
    :param nu:
    :return: SABR volatility
    """
    f = s * np.exp((r - q * tow))

    z = nu / alp * (f * K) ** (0.5 * (1 - bet)) * np.log(f / K)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    zdivxz = z / xz

    # exception cases
    zdivxz[np.isnan(zdivxz)] = 1.0
    result = (alp * (f * K) ** (0.5 * (bet - 1)) *
              (1 + (((1 - bet) * np.log(f / K)) ** 2 / 24 + ((1 - bet) * np.log(f / K)) ** 4 / 1920)) ** (-1.0)
              * zdivxz
              * (1 + (((1 - bet) * alp) ** 2 / (24 * (f * K) ** (1 - bet))
                      + 0.25 * alp * bet * rho * nu / ((f * K) ** (0.5 * (1 - bet)))
                      + ((2 - 3 * rho ** 2) * nu ** 2) / 24) * tow)
              )
    return result
