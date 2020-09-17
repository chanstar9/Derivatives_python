# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 
"""
from copy import deepcopy
import numpy as np

a = np.array([3., 1, 3])
b = np.array([10., 10., 7., 4.])
c = np.array([2., 4., 5.])
d = np.array([[3, 4, 5, 6.], [4, 5, 6., 7], [5, 6., 7, 8]])


def TDMAsolver1(aa, bb, cc, dd):
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


def TDMAsolver(aa, bb, cc, dd):
    nf = len(dd)  # number of edivuations
    ac, bc, cc, dc = map(np.array, (aa, bb, cc, dd))  # copy the array
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in reversed(range(0, nf - 1)):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


for i in range(3):
    print(TDMAsolver(a, b, c, d[i]))

TDMAsolver1(a, b, c, d)

aa = a
bb = b
cc = c
dd = d[0]
dd = d
