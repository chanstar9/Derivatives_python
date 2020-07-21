# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 06
"""
from keras.backend import sigmoid


def swish(x, beta=0.5):
    return x * sigmoid(beta * x)


