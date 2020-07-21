# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 04:20:22 2019

@author: ???
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import style
import datetime as dt
import os

path = os.getcwd()

style.use('ggplot')


# interpolation formula
def interpolation(start, date1, rate1, date2, rate2, target_date):
    x1 = (date1 - start).days
    x2 = (date2 - start).days
    x_target = (target_date - start).days
    target = (x_target - x1) * (rate2 - rate1) / (x2 - x1) + rate1
    return target


# %%
# parameter setting
# get historical data from yahoo finance
start = dt.datetime(2008, 4, 3)
end = dt.datetime(2019, 4, 3)

pricing_date = dt.date(2019, 9, 6)
maturity_date = dt.date(2022, 9, 2)
rho = 0.6212
# rho = 0.92

r_1 = 0.01378  # US IRS
r_2 = -0.005304  # EURO IRS
stx_div = 0
rty_div = 0
stx0 = 2976
rty0 = 3484.70
n_trials = 10000  # 시뮬 횟수
T = 3
FV = 10000
n_steps = 600
# avg_node = [(avg_start - pricing_date).days, (avg_end - pricing_date).days]
# estimate sigma
stx_sigma = 0.1570
rty_sigma = 0.1465
K = [90, 90, 90, 85, 80, 75]
c = 0.018 * np.array(range(1, T * 2 + 1))
barrier_rate = 0.5
knock_in = 0


# %%
def project3_sim(stx0, rty0, r, stx_div, rty_div, stx_sigma, rty_sigma, rho, T, FV, n_steps, n_trials):
    maximum_value = FV * 2
    d_t = T / n_steps
    # avg_start_node = avg_node[0]
    # avg_end_node = avg_node[1]
    z_matrix_stx = np.random.standard_normal((n_trials, n_steps))
    z_matrix_rty = rho * np.random.standard_normal((n_trials, n_steps)) + np.sqrt(1 - rho ** 2) * np.random.standard_normal((n_trials, n_steps))
    stx_matrix = np.zeros((n_trials, n_steps))
    rty_matrix = np.zeros((n_trials, n_steps))
    stx_matrix[:, 0] = stx0
    rty_matrix[:, 0] = rty0
    for j in range(n_steps - 1):
        stx_matrix[:, j + 1] = stx_matrix[:, j] * np.exp(
            (r_1 - stx_div - 0.5 * stx_sigma ** 2) * d_t + stx_sigma * np.sqrt(d_t) * z_matrix_stx[:, j])
        rty_matrix[:, j + 1] = rty_matrix[:, j] * np.exp(
            (r_2 - rty_div - 0.5 * rty_sigma ** 2) * d_t + rty_sigma * np.sqrt(d_t) * z_matrix_rty[:, j])

    payoff = np.zeros(n_trials, 600)

    for i in tqdm(range(n_trials)):
        if avg_stx[i] >= 1.21 * stx0 and avg_rty[i] >= 1.21 * rty0:
            payoff[i] = min(FV + FV * (min(avg_stx[i] / stx0, avg_rty[i] / rty0) - 1.21) * 3.34 + 415, maximum_value)
        elif avg_stx[i] < 1.21 * stx0 or avg_rty[i] < 1.21 * rty0:
            if avg_stx[i] >= stx0 and avg_rty[i] >= rty0:
                payoff[i] = FV + FV * (min(avg_stx[i] / stx0, avg_rty[i] / rty0) - 1) * 1.5 + 100
            elif avg_stx[i] < stx0 or avg_rty[i] < rty0:
                if avg_stx[i] >= 0.95 * stx0 and avg_rty[i] >= 0.95 * rty0:
                    payoff[i] = FV + FV * (min(avg_stx[i] / stx0, avg_rty[i] / rty0) - 0.95) * 2
                elif avg_stx[i] < 0.95 * stx0 or avg_rty[i] < 0.95 * rty0:
                    payoff[i] = FV * min(avg_stx[i] / stx0, avg_rty[i] / rty0) + 50
    value = np.mean(payoff) * np.exp(-r * T)

    return value


# %%
price = project3_sim(stx0, rty0, r, stx_div, rty_div, stx_sigma, rty_sigma, rho, T, FV, avg_node, n_steps, n_trials)
print('-' * 50)
print('price is %4.3f' % (price))
print('-' * 50)
# %%
# =============================================================================
# Sensitivity analysis
# =============================================================================

# 1. depending on correlation
print('price depending on rho change')
rho_range = np.arange(-1, 1.01, 0.1)
rho_price = []
for est_rho in rho_range:
    est_pice = project3_sim(stx0, rty0, r, stx_div, rty_div, stx_sigma, rty_sigma, est_rho, T, FV, avg_node, n_steps,
                            n_trials)
    rho_price.append(est_pice)
# %%
plt.figure(0)
plt.plot(rho_range, rho_price, label='price')
plt.scatter(rho, price, label='current price', color='b')
plt.title('price depending on rho change')
plt.ylabel('price')
plt.xlabel('rho')
plt.legend()
# %%
print('price depending on stx dividend change')
# 2.depending on dividend yield
div_range = np.arange(0, 0.1, 0.01)
div_price = []
for est_div in div_range:
    est_price = project3_sim(stx0, rty0, r, est_div, rty_div, stx_sigma, rty_sigma, rho, T, FV, avg_node, n_steps,
                             n_trials)
    div_price.append(est_price)
    # %%
plt.figure(1)
plt.plot(div_range, div_price, label='price')
plt.scatter(stx_div, price, label='current price', color='b')
plt.title('price depending on stx dividend change')
plt.ylabel('price')
plt.xlabel('dividend yield')
plt.legend()
# %%
# 3depending on Volatility
print('price depending on stx volatility change')
vol_range = np.arange(0, 0.5, 0.05)
vol_price = []
for est_vol in vol_range:
    est_price = project3_sim(stx0, rty0, r, stx_div, rty_div, est_vol, rty_sigma, rho, T, FV, avg_node, n_steps,
                             n_trials)
    vol_price.append(est_price)
# %%
plt.figure(2)
plt.plot(vol_range, vol_price, label='price')
plt.scatter(stx_sigma, price, label='current price', color='b')
plt.title('price depending on stx volatility change')
plt.ylabel('price')
plt.xlabel('standard deviation')
plt.legend()
# %%
# 4.depending on risk free rate
print('price depending on risk free rate change')
r_range = np.arange(0, 0.09, 0.01)
r_price = []
for est_r in r_range:
    est_price = project3_sim(stx0, rty0, est_r, stx_div, rty_div, stx_sigma, rty_sigma, rho, T, FV, avg_node, n_steps,
                             n_trials)
    r_price.append(est_price)

plt.figure(3)
plt.plot(r_range, r_price, label='price')
plt.scatter(r, price, label='current price', color='b')
plt.title('price depending on risk free rate change')
plt.ylabel('price')
plt.xlabel('risk free rate')
plt.legend()
# %%
print('price depending on the number of simulation change')
# trial_range = range(1000,10000,1000)
# trials_price = []
# for est_trials in trial_range:
#    est_price = project3_sim(stx0,rty0,r,stx_div,rty_div,stx_sigma,rty_sigma,rho,T,FV,avg_node,n_steps,est_trials)
#    trials_price.append(est_price)
#    
# plt.figure(4)
# plt.plot(trial_range,trials_price,label = 'price')
# plt.scatter(n_trials,price,label = 'current price',color='b')
# plt.title('price depending on the number of simulation change')
# plt.ylabel('price')
# plt.xlabel('the number of simulation')
# plt.legend()
