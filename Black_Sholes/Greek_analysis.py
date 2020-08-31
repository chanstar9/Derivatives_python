# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 07. 22
"""
from Black_Sholes.bs_formula import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

underlying_scenario = np.arange(1, 100, 2)
k = 50
tows = np.arange(1/365, 3, 0.01)
r = 0.01
q = 0
vols = [0.3, 0.7]
position = 1

# delta
delta = pd.DataFrame(index=underlying_scenario)
for j in tows:
    lst = []
    for i in underlying_scenario:
        lst.append(bs_delta(i, k, j, r, q, vols[0], position, cp=-1))
    delta = pd.concat([delta, pd.DataFrame(lst, index=underlying_scenario)], axis=1)

delta.columns = tows

fig = plt.figure()
ax = fig.gca(projection='3d')
xnew, ynew = np.meshgrid(underlying_scenario, tows)
surface = ax.plot_surface(xnew, ynew, delta.values.T, cmap=cm.winter)
ax.set_xlabel('S')
ax.set_ylabel('Tow')
plt.title('Delta')
plt.show()

# gamma
gamma = pd.DataFrame(index=underlying_scenario)
for j in tows:
    lst = []
    for i in underlying_scenario:
        lst.append(bs_gamma(i, k, j, r, q, vols[0], position))
    gamma = pd.concat([gamma, pd.DataFrame(lst, index=underlying_scenario)], axis=1)

gamma.columns = tows

fig = plt.figure()
ax = fig.gca(projection='3d')
xnew, ynew = np.meshgrid(underlying_scenario, tows)
surface = ax.plot_surface(xnew, ynew, gamma.values.T, cmap=cm.winter)
ax.set_xlabel('S')
ax.set_ylabel('Tow')
plt.title('gamma')
plt.show()

# theta
theta = pd.DataFrame(index=underlying_scenario)
for j in tows:
    lst = []
    for i in underlying_scenario:
        lst.append(bs_theta(i, k, j, r, q, vols[0], position, cp=-1))
    theta = pd.concat([theta, pd.DataFrame(lst, index=underlying_scenario)], axis=1)

theta.columns = tows

fig = plt.figure()
ax = fig.gca(projection='3d')
xnew, ynew = np.meshgrid(underlying_scenario, tows)
surface = ax.plot_surface(xnew, ynew, theta.values.T, cmap=cm.winter)
ax.set_xlabel('S')
ax.set_ylabel('Tow')
plt.title('theta')
plt.show()

# vega
vega = pd.DataFrame(index=underlying_scenario)
for j in tows:
    lst = []
    for i in underlying_scenario:
        lst.append(bs_vega(i, k, j, r, q, vols[0], position, cp=-1))
    vega = pd.concat([vega, pd.DataFrame(lst, index=underlying_scenario)], axis=1)

vega.columns = tows

fig = plt.figure()
ax = fig.gca(projection='3d')
xnew, ynew = np.meshgrid(underlying_scenario, tows)
surface = ax.plot_surface(xnew, ynew, vega.values.T, cmap=cm.winter)
ax.set_xlabel('S')
ax.set_ylabel('Tow')
plt.title('Vega')
plt.show()


# delta
delta_df = pd.DataFrame(index=underlying_scenario)
for vol in vols:
    lst = []
    for j in underlying_scenario:
        lst.append(bs_delta(j, k, tows[1], r, q, vol, position))
    delta_df = pd.concat([delta_df, pd.DataFrame(lst, index=underlying_scenario)], axis=1)

delta_df.columns = vols
delta_df.plot()
