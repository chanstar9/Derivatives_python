# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 
"""
import numpy as np


def call(s, k):
    return np.max(s - k, 0)


def put(s, k):
    return np.max(k - s, 0)
