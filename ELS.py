# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 02. 06
"""

from copy import deepcopy as dc
from datetime import datetime

import numpy as np
from numpy import array

from columns import *
from stock_process import GBM


class ELS:
    """
    To get ELS price, greek
    """

    def __init__(self, underlyings, start_date: datetime, F, redemption_dates: array, coupon: list, K: list,
                 barrier_rate: float, T: float):
        if underlyings not in INDICES:
            raise ValueError('{} is not registered.'.format(underlyings))
        self.start_date = start_date
        self.redemption_dates = redemption_dates
        self.coupon = dict(zip(redemption_dates, coupon))
        self.K = dict(zip(redemption_dates, K))
        self.barrier_rate = barrier_rate
        self.T = T

    def get_price(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        s = GBM(s0=s0, mean=mean, cov=cov, iter_num=iter_num, N=N, T=self.T, div=div, seed=seed)

        return price

    def get_greek(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        greeks = {
            DELTA: self.get_delta(s0=s0, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            GAMMA: self.get_gamma(s0=s0, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            THETA: self.get_theta(s0=s0, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            VEGA: self.get_vega(s0=s0, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                exchange_rate=exchange_rate)
        }
        return greeks

    def get_delta(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        pu = self.get_price(s0=s0 * 1.01, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                            exchange_rate=exchange_rate)
        pd = self.get_price(s0=s0 * 0.99, mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                            exchange_rate=exchange_rate)
        return (pu - pd) / 0.02

    def get_gamma(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return

    def get_vega(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return

    def get_theta(self, s0, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return


if __name__ == '__main__':
    # example
    redemption_dates = np.array(
        [datetime(2020, 3, 5), datetime(2020, 9, 4), datetime(2021, 3, 5), datetime(2021, 9, 3),
         datetime(2022, 3, 4), datetime(2022, 9, 2)])
    coupon = [0.018, 0.036, 0.054, 0.072, 0.09, 0.108]
    K = [0.9, 0.9, 0.9, 0.85, 0.8, 0.75]
    barrier_rate = 0.5
    T = 3
    self = ELS(redemption_dates=redemption_dates, coupon=coupon, K=K, barrier_rate=barrier_rate, T=T)
    mean = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    div = np.array([0, 0])
