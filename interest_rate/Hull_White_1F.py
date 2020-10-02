# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 21
"""
import QuantLib as ql
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

sigma = 0.01
a = 0.001
timestep = 360
length = 30  # in years
forward_rate = 0.05
day_count = ql.Thirty360()
todays_date = ql.Date(15, 1, 2015)

ql.Settings.instance().evaluationDate = todays_date

yield_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
spot_curve_handle = ql.YieldTermStructureHandle(yield_curve)

hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(
        timestep,
        ql.UniformRandomGenerator(125)))
seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)


def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep + 1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr


num_paths = 128
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Hull-White Short Rate Simulation")
plt.show()
