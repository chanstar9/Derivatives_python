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
    :param 0 <= alp: control the height of the ATM implied volatility level
    :param 0 <= bet <= 1: controls curvature
    :param -1 <= rho <= 1: instantaneous correlation between underlying and its volatility
    :param nu:
    :return:
    """
    f = s * np.exp((r - q * tow))


    return
