# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 23
"""
import pandas as pd
import numpy as np
from Black_Sholes import bs_formula as bs
from datetime import datetime
from math import exp
import matplotlib.pyplot as plt
from pylab import cm
from scipy.optimize import curve_fit
from scipy import interpolate
from volatility import SABR as sa


def time_tango(dates):
    return datetime.strptime("{}".format(dates), "%Y-%m-%d")


convDates = 365.0  # date convention
V0 = 2952.48  # EUROSTOXX50 index at '2016-01-15'
qd = '2016-01-15'  # quote date
cc = 0.015  # example cost of carry

# excel file read
xl = pd.ExcelFile("data/option/20160101_EUROSTOXX50.xlsx")

# select sheet
euro = xl.parse("Euro")

# make a float type
euro['STRIKE'] *= 1.0

# set column name
euro['IMVOL'] = 0.0

# calculate time to maturity
euro['TTM'] = euro.apply(lambda x: 1.0 * ((time_tango(x['EXP_DATE'])) - time_tango(qd)).days / convDates, axis=1)


# find a unique value (remove duplicate)
expirySet = pd.Series(euro['EXP_DATE'].values.ravel()).unique()
stkSet = pd.Series(euro['STRIKE'].values.ravel()).unique()

# init
y1 = np.zeros([len(expirySet), len(stkSet)])
ttm = np.zeros(len(expirySet))

# time slices iteration
for it in range(len(expirySet)):
    # iterate time(maturity) slice
    expiry = expirySet[it]

    # select call
    sc = euro.loc[(euro['QUOTE_DATE'] == qd) & (euro['OPTION_TYPE'] == 'Call')]
    # select put
    sp = euro.loc[(euro['QUOTE_DATE'] == qd) & (euro['OPTION_TYPE'] == 'Put')]

    # time to maturity
    ttm[it] = 1.0 * (time_tango(expiry) - time_tango(qd)).days / convDates

    # select time slice
    sc1 = sc.loc[sc['EXP_DATE'] == expiry]  # call
    sp1 = sp.loc[sp['EXP_DATE'] == expiry]  # put

    stkcp = list(set(sc1['STRIKE']) & set(sp1['STRIKE']))

    # init, tmpcp: stk | CALL price | PUT price
    tmpcp = np.zeros((len(stkcp), 3))

    for i in range(len(stkcp)):
        tmpcp[i, 0] = stkcp[i]
        tmpcp[i, 1] = sc1[sc1['STRIKE'] == stkcp[i]]['PRICE']
        tmpcp[i, 2] = sp1[sp1['STRIKE'] == stkcp[i]]['PRICE']

    dif = abs(tmpcp[:, 1] - tmpcp[:, 2])  # difference

    # minimum ((near the) At The Money)
    mindif = np.min(dif)
    minK = tmpcp[dif == mindif, 0]
    minKIdx = np.where(tmpcp[:, 0] == minK)[0][0]

    # synthetic forward(S = C-P + Ke^(-cc*T)) by using Put-Call parity
    mV0 = tmpcp[minKIdx, 1] - tmpcp[minKIdx, 2] + tmpcp[minKIdx, 0] * exp(-cc * ttm[it])

    # S0 = dataC(minIdx,2)-dataP(minIdx,2)...
    #    +dataC(minIdx,1)*exp(-(r-q)*Tau); % No-arbitrage condition(P-C parity)

    # find a implied vol (using Newton method)
    for i in sc1.index:
        imp_vol = bs.IV(mV0, sc1.loc[i]['STRIKE'], sc1.loc[i]['TTM'], cc, 0, sc1.loc[i]['PRICE'], 1)
        sc1.loc[i]['IMVOL'] = imp_vol
    for i in sp1.index:
        imp_vol = bs.IV(mV0, sp1.loc[i]['STRIKE'], sp1.loc[i]['TTM'], cc, 0, sp1.loc[i]['PRICE'], -1)
        sp1.loc[i]['IMVOL'] = imp_vol

    # By using synthetic forward, combine implied vol of OTM put and call
    lenc = sum(sc1['STRIKE'] <= float(minK))
    lenp = sum(sp1['STRIKE'] > float(minK))

    # init
    cp = np.zeros((2, lenc + lenp))

    # write the obtained solution
    cp[0, :lenc] = sc1[sc1['STRIKE'] <= float(minK)]['STRIKE']
    cp[0, lenc:] = sp1[sp1['STRIKE'] > float(minK)]['STRIKE']
    cp[1, :lenc] = sc1[sc1['STRIKE'] <= float(minK)]['IMVOL']
    cp[1, lenc:] = sp1[sp1['STRIKE'] > float(minK)]['IMVOL']

    # trick for using some variables in function like global variables
    sa.SetVar(mV0, ttm[it], cc, 0.0)
    # y = sa.SABR_func(cp[0, :], 0.8, 0.9, 0.3, 0.15);

    # popt, pcov = curve_fit(func, x, yn)
    init_guess = np.array([0.5, 0.5, 0.5, 0.5])  # initial value
    sa.SetVar(mV0, ttm[it], cc, 0.0)

    # Least square curve fit by Scipy
    # By using this function, fit implied vol curve to SABR function at one of time slices
    popt, pcov = curve_fit(sa.SABR_vol, cp[0, :], cp[1, :], p0=init_guess)

    # write
    y1[it, :] = sa.SABR_vol(stkSet, *popt)

    print("iteration %d done..." % it)


# Plot implied vol surface fitted SABR
K, T = np.meshgrid(stkSet, ttm)
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(K, T, y1, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('maturity')
ax.set_zlabel('Fitting vol')
plt.title('Implied vol surface fitted SABR')

# interpolation 2D is applied at implied vol suface.
# Since time slices are nonuniform,
# I think that this gives a drawback for differencing local vol
# Therefore, I employ 2D interpolation.
# Notice that I choose 'cubic', however you can choose 'linear'
# cubic interpolation
intplf = interpolate.interp2d(stkSet, ttm, y1, kind='cubic')
num = 50  # # of grids
newStk = np.linspace(min(stkSet), max(stkSet), num)
newTtm = np.linspace(min(ttm), max(ttm), num)
y2 = intplf(newStk, newTtm)

newK, newT = np.meshgrid(newStk, newTtm)

# Plot interpolated implied vol surface fitted SABR
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(newK, newT, y2, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('maturity')
ax.set_zlabel('Fitting vol')
plt.title('Interpolated implied vol surface fitted SABR')

# init
dx = np.zeros_like(y2)
dT = np.zeros_like(y2)
dxx = np.zeros_like(y2)
d = np.zeros_like(y2)

h = np.diff(newStk)  # spatial steps
k = np.diff(newTtm)  # temporal steps
# find derivatives
for i in range(1, len(newTtm) - 1):
    for j in range(1, len(newStk) - 1):
        dx[i, j] = (-h[j] / (h[j - 1] * (h[j - 1] + h[j])) * y2[i, j - 1]
                    + (h[j] - h[j - 1]) / (h[j - 1] * h[j]) * y2[i, j]
                    + h[j - 1] / (h[j] * (h[j - 1] + h[j])) * y2[i, j + 1])
        dxx[i, j] = 2.0 * (y2[i, j - 1] / (h[j - 1] * (h[j - 1] + h[j]))
                           - y2[i, j] / (h[j - 1] * h[j])
                           + y2[i, j + 1] / (h[j] * (h[j - 1] + h[j])))
        dT[i, j] = (-k[i] / (k[i - 1] * (k[i - 1] + k[i])) * y2[i - 1, j]
                    + (k[i] - k[i - 1]) / (k[i - 1] * k[i]) * y2[i, j]
                    + k[i - 1] / (k[i] * (k[i - 1] + k[i])) * y2[i + 1, j])
        d[i, j] = ((np.log(mV0 / newStk[j]) + (cc + 0.5 * y2[i, j] ** 2) *
                    newTtm[i]) / (y2[i, j] * np.sqrt(newTtm[i])))
# init
lovol = np.zeros_like(y2)

# By Dupire equation,
for i in range(len(newTtm)):
    for j in range(len(newStk)):
        lovol[i, j] = ((y2[i, j] ** 2 + 2.0 * y2[i, j] * newTtm[i] *
                        (dT[i, j] + (cc) * newStk[j] * dx[i, j])) / ((1.0 + newStk[j] * d[i, j] * dx[i, j] *
                                                                      np.sqrt(newTtm[i])) ** 2 + y2[i, j] * newStk[
                                                                         j] ** 2 * newTtm[i] * (dxx[i, j] - d[i, j] *
                                                                                                dx[i, j] ** 2 * np.sqrt(
                    newTtm[i]))))

# linear extrapolation at every boundary
lovol[:, 0] = 2.0 * lovol[:, 1] - lovol[:, 2]
lovol[:, -1] = 2.0 * lovol[:, -2] - lovol[:, -3]
lovol[0, :] = 2.0 * lovol[1, :] - lovol[2, :]
lovol[-1, :] = 2.0 * lovol[-2, :] - lovol[-3, :]
lovol = np.sqrt(lovol)

# plot local vol surface
fig = plt.figure(figsize=(12, 7))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(newK, newT, lovol, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('maturity')
ax.set_zlabel('local vol')
plt.title('Local vol surface')
