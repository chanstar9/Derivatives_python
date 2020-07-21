# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 02. 05
"""

import numpy as np


def mc_simulation(path, derivatives, r):
    payoffs = []
    for _ in range(len(path)):
        payoffs.append(derivatives(path))
    discount_factors = np.exp(-r * len(path[0]))
    price = discount_factors * (sum(payoffs) / len(path))
    return price
