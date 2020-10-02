# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 09. 15
"""
from scipy.stats import norm
import numpy as np


def d_1(s, k, tow, r, q, vol):
    return (np.log(s / k) + (r - q + (vol ** 2) * 0.5) * tow) / (vol * np.sqrt(tow))


def bs_price(s, k, tow, r, q, vol, position, cp=1):
    d1 = d_1(s, k, tow, r, q, vol)
    d2 = d1 - vol * np.sqrt(tow)
    return position * (cp * s * np.exp(-q * tow) * norm.cdf(cp * d1) - cp * k * np.exp(-r * tow) * norm.cdf(cp * d2))


def bs_delta(s, k, tow, r, q, vol, position, cp=1):
    d1 = d_1(s, k, tow, r, q, vol)
    return position * (cp * np.exp(-q * tow) * norm.cdf(cp * d1))


def bs_vega(s, k, tow, r, q, vol, position):
    d1 = d_1(s, k, tow, r, q, vol)
    return position * (s * np.exp(-q * tow) * np.sqrt(tow) * norm.pdf(d1))


def bs_gamma(s, k, tow, r, q, vol, position):
    d1 = d_1(s, k, tow, r, q, vol)
    return position * (np.exp(-q * tow) * norm.pdf(d1) / (s * vol * np.sqrt(tow)))


def bs_theta(s, k, tow, r, q, vol, position, cp=1):
    d1 = d_1(s, k, tow, r, q, vol)
    d2 = d1 - vol * np.sqrt(tow)
    return position * (-np.exp(-q * tow) * s * norm.pdf(d1) * vol / (2 * np.sqrt(tow)) - cp * r * k * np.exp(
        -r * tow) * norm.cdf(cp * d2) + cp * q * s * np.exp(-q * tow) * norm.cdf(cp * d1))


def bs_rho(s, k, tow, r, q, vol, position, cp=1):
    d1 = d_1(s, k, tow, r, q, vol)
    d2 = d1 - vol * np.sqrt(tow)
    return position * (cp * k * tow * np.exp(-r * tow) * norm.cdf(cp * d2))


def IV(s, k, tow, r, q, mkt_price, cp=1):
    global price
    vol = 0.15
    while abs(price - mkt_price) >= 1.0e-6:
        price = bs_price(s, k, tow, r, q, vol, cp)
        vol = vol - (price - mkt_price) / bs_vega(s, k, tow, r, q, vol, 1)
    return vol
