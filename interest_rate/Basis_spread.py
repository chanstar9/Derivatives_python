# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 10. 10
"""
from columns import *

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

LIBOR_TOTAL = pd.ExcelFile('data/interest_rate/LIBOR_TOTAL.xlsx')
basis = LIBOR_TOTAL.parse(BASIS_SWAP_RATE).dropna()
basis.set_index('Timestamp', inplace=True)
basis.sort_index(inplace=True)
basis_col = [col for col in basis.columns if '6L' in col]
basis = basis[basis_col]
basis_col = [int(col.split('=')[0][7:][:-1]) for col in basis.columns]

X = np.arange(0, 2394)
Y = basis_col
Z = basis.values
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z.T, cmap=cm.jet)
ax.set_xlabel('time')
ax.set_ylabel('tenor')
ax.set_zlabel('basis swap rate')
plt.title('6M-3M LIBOR')
plt.show()

print(np.shape(X), np.shape(Y), np.shape(Z.T))
