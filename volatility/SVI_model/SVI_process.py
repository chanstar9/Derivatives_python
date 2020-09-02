from __future__ import division

import pandas as pd
import scipy as sp

from volatility.SVI_model.cal_func import *


# Do logging to stdout
def log(*ss):
    for s in ss: print(s)


# Define some constants
S0 = 309  # KS200 index
q = 0.02335  # dividend(6m avg)
r = 0.0079  # interest rate(CD91)
sig0 = 0.24  # initial volatility guess
lvol = 0.03  # lower volatility acceptance limit
uvol = 3.00  # upper volatility acceptance limit
bpen = 128  # initial butterfly penalty factor
cpen = 128  # initial calendar penalty factor
blim = 0.001  # target butterfly arbitrage bound
clim = 0.001  # target calendar arbitrage bound

# Read raw data
log('Reading raw data ...')
data = pd.read_csv('data/KS200_option_data.csv')  # raw data from csv
num = pd.DataFrame(index=sorted(set(data['Expiration'])))
num['Raw'] = [len(data.loc[data['Expiration'] == T]) for T in sorted(set(data['Expiration']))]
data['LogStrike'] = [logstrike(K, T, S0, r, q) for K, T in zip(data['Strike'], data['Expiration'])]

log('Calculating implied volatilities ...')
data['IV'] = [sp.optimize.bisect(bsaux, -1, 100, args=("C", S0, K, r, T, q, mid_price), xtol=1e-3) for K, T, mid_price
              in zip(data['Strike'], data['Expiration'], data['Mid_Matrix'])]
data['Bid_IV'] = [sp.optimize.bisect(bsaux, -1, 100, args=("C", S0, K, r, T, q, mid_price), xtol=1e-3) for
                  K, T, mid_price in zip(data['Strike'], data['Expiration'], data['Bid_price'])]
data['Ask_IV'] = [sp.optimize.bisect(bsaux, -1, 100, args=("C", S0, K, r, T, q, mid_price), xtol=1e-3) for
                  K, T, mid_price in zip(data['Strike'], data['Expiration'], data['Ask_price'])]

# Clean raw data wrt an implied volatility bound and report number of records
log('Cleaning data to ensure ' + str(lvol) + ' <= IV <= ' + str(uvol) + ' ...')
data = data.loc[(data['IV'] > lvol) & (data['IV'] < uvol), :]
num['Clean'] = [len(data.loc[data['Expiration'] == T]) for T in sorted(set(data['Expiration']))]
log('Number of records in raw and cleaned dataset:', num)

# Prepare grid on which to check presence of arbitrage
expirs = sorted(set(data['Expiration']))
strikes = sorted(set(data['Strike']))
grid = pd.DataFrame(index=strikes)
for T in expirs: grid[T] = [logstrike(K, T, S0, r, q) for K in strikes]

# Variable to store parameter vectors chi
chi = pd.DataFrame(index=expirs, columns=['m1', 'm2', 'q1', 'q2', 'c'])


# Residuals function for fitting implied volatility
def residSVI(chi, T):
    w = [straightSVI(k, chi[0], chi[1], chi[2], chi[3], chi[4]) for k in data.loc[data['Expiration'] == T, 'LogStrike']]
    return data.loc[data['Expiration'] == T, 'IV'] - np.sqrt(np.array(w) / T)


# Function to obtain initial parameter vector for fit
def chi0(T):
    # Split data in five intervals and calculate mean x and mean y
    kmin = np.min(data.loc[data['Expiration'] == T, 'LogStrike'])
    kmax = np.max(data.loc[data['Expiration'] == T, 'LogStrike'])
    klo = [kmin + i * (kmax - kmin) / 5. for i in range(5)]
    kup = [kmin + (i + 1) * (kmax - kmin) / 5. for i in range(5)]
    xm = np.array(
        [np.mean(data.loc[(data['Expiration'] == T) & (l <= data['LogStrike']) & (data['LogStrike'] <= u), 'LogStrike'])
         for l, u in zip(klo, kup)])
    ym = np.array([np.mean(
        T * data.loc[(data['Expiration'] == T) & (l <= data['LogStrike']) & (data['LogStrike'] <= u), 'IV'] ** 2) for
        l, u in zip(klo, kup)])

    # Determine quadratic form through these five average points
    un = np.array([1] * len(klo))
    A = np.matrix([ym * ym, ym * xm, xm * xm, ym, un]).T
    a = np.linalg.solve(A, -xm)

    # If it's already a hyperbola, we have our initial guess
    if 4 * a[0] * a[2] < a[1] ** 2: return std2straight(a)

    # Otherwise, flip to approximating hyperbola and do a least squares fit to the five points
    a[0] = -a[0]

    def residHyp(chi):
        return np.array([straightSVI(x, chi[0], chi[1], chi[2], chi[3], chi[4]) for x in xm]) - ym

    ap = sp.optimize.leastsq(residHyp, std2straight(a))
    return ap[0]


# Fit implied volatilities directly to obtain first guess on parameter vectors
i = 0
log('Calculating first guess on parameters ...')
for T in expirs:
    i += 1
    log('Fitting implied volatility on slice ' + str(i) + ', T=' + str(T) + ' ...')
    chi.loc[T, :] = chi0(T)
    chi.loc[T, :] = sp.optimize.leastsq(residSVI, list(chi.loc[T, :]), args=(T))[0]
    log('Got parameters:', chi.loc[T, :])
log('Summary of initial guess for parameters:', chi)


# Function to quantify calendar arbitrage between two slices T1 > T2 on grid
def calendar(chi1, T1, chi2, T2):
    if T2 == 0 or T1 <= T2: return 0
    w1 = [straightSVI(k, chi1[0], chi1[1], chi1[2], chi1[3], chi1[4]) for k in grid[T1]]
    w2 = [straightSVI(k, chi2[0], chi2[1], chi2[2], chi2[3], chi2[4]) for k in grid[T2]]
    return sum([np.maximum(0, x2 - x1) for x1, x2 in zip(w1, w2)])


# Function to quantify butterfly arbitrage in a slice on grid
def butterfly(chi, T):
    w = np.array([straightSVI(k, chi[0], chi[1], chi[2], chi[3], chi[4]) for k in grid[T]])
    wp = np.array([straightSVIp(k, chi[0], chi[1], chi[2], chi[3], chi[4]) for k in grid[T]])
    wpp = np.array([straightSVIpp(k, chi[0], chi[1], chi[2], chi[3], chi[4]) for k in grid[T]])
    g = (1. - (grid[T] * wp) / (2. * w)) ** 2 - wp ** 2 / 4. * (1. / w + 1. / 4.) + wpp / 2.
    return sum([np.maximum(0, -x) for x in g])


# Residuals function for fitting option prices with penalties on arbitrage
def residuals(chiT, T, Tp):
    w = [straightSVI(k, chiT[0], chiT[1], chiT[2], chiT[3], chiT[4]) for k in
         data.loc[data['Expiration'] == T, 'LogStrike']]
    bs = [BlackScholes("C", S0, K, r, sig, T, q) for K, sig in
          zip(data.loc[data['Expiration'] == T, 'Strike'], np.sqrt(np.array(w) / T))]
    calarbT = calendar(chiT, T, chi.loc[Tp, :], Tp) if Tp else 0
    butarbT = butterfly(chiT, T)
    e = data.loc[data['Expiration'] == T, 'Mid_Matrix'] - bs
    return e + (np.sqrt(sum(e) ** 2 + (cpen * calarbT + bpen * butarbT) ** 2 * len(e)) - sum(e)) / len(e)


# Reduce arbitrage by fitting option prices with penalties on calendar and butterfly arbitrage
maxbutarb = float("Inf")
maxcalarb = float("Inf")
while maxbutarb > blim or maxcalarb > clim:
    log('Butterfly penalty factor: ' + str(bpen))
    log('Calendar penalty factor: ' + str(cpen))
    j = 0
    Tp = 0
    maxbutarb = 0
    maxcalarb = 0
    for T in expirs:
        j += 1
        log('Fitting mid prices on slice ' + str(j) + ', T=' + str(T) + ' ...')
        chi.loc[T, :] = sp.optimize.leastsq(residuals, list(chi.loc[T, :]), args=(T, Tp))[0]
        log('Got parameters:', chi.loc[T, :])
        butarb = butterfly(chi.loc[T, :], T)
        log('Butterfly penalty for slice is ' + str(bpen * butarb))
        calarb = calendar(chi.loc[T, :], T, chi.loc[Tp, :], Tp) if Tp else 0
        log('Calendar penalty for slice is ' + str(cpen * calarb))
        maxbutarb = np.maximum(maxbutarb, butarb)
        maxcalarb = np.maximum(maxcalarb, calarb)
        Tp = T
    if maxbutarb > clim: bpen *= 2
    if maxcalarb > clim: cpen *= 2

log('Maximum remaining butterfly arbitrage is ' + str(maxbutarb))
log('Maximum remaining calendar arbitrage is ' + str(maxcalarb))
log('Summary of final parameters:', chi)

#
# Report raw parameters and draw plots with final fit
#
raw = pd.DataFrame(index=expirs, columns=['a', 'b', 'rho', 'm', 'sigma'])
for T in expirs: raw.loc[T, :] = straight2raw(chi.loc[T, :])
log('Final raw SVI parameters:', raw)
doplots('KS200_SVI123', expirs, data, chi, grid, S0, r, q)

# def discriminant(a,b,c,d,e):
#    return 256*a**3*e**3-192*a**2*b*d*e**2-128*a**2*c**2*e**2 +144*a**2*c*d**2*e-27*a**2*d**4\
#           + 144*a*b**2*c*e**2 - 6*a*b**2*d**2*e -80*a*b*c**2*d*e+18*a*b*c*d**3+16*a*c**4*e\
#           -4*a*c**3*d**2-27*b**4*e**2+18*b**3*c*d*e-4*b**3*d**3-4*b**2*c**3*e+b**2*c**2*d**2
