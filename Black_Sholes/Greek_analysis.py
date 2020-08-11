# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 07. 22
"""
from Black_Sholes.bs_formula import *
import pandas as pd
import matplotlib.pyplot as plt

s = np.arange(1, 100, 2)
k = 50
tows = [0.2, 0.5, 1]
r = 0.01
q = 0
vols = [0.3, 0.5]
position = 1

# vega
vega = pd.DataFrame(index=s)
for j in tows:
    lst = []
    for i in s:
        lst.append(bs_vega(i, k, j, r, q, vols[0], position))
    vega = pd.concat([vega, pd.DataFrame(lst, index=s)], axis=1)

vega.columns = tows
vega.plot()

# delta
delta_df = pd.DataFrame(index=s)
for vol in vols:
    lst = []
    for j in s:
        lst.append(bs_delta(j, k, tows[1], r, q, vol, position))
    delta_df = pd.concat([delta_df, pd.DataFrame(lst, index=s)], axis=1)

delta_df.columns = vols
delta_df.plot()
