# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 02. 06
"""

from datetime import datetime, timedelta
from tqdm import tqdm

from numpy import array

from columns import *
from stock_process import GBM
from simulation_method.FDM.cal_func import *


class ELS:
    """
    To get ELS price, greek
    """

    def __init__(self, underlyings, initial_underlyings, start_date: datetime, F, redemption_dates: array, coupon: list,
                 K: list, barrier_rate: float, T: float):
        for underlying in underlyings:
            if underlying not in INDICES:
                raise ValueError('{} is not registered.'.format(underlying))
        self.initial_underlyings = initial_underlyings
        self.start_date = start_date
        self.redemption_dates = redemption_dates
        self.coupon = dict(zip(redemption_dates, coupon))
        self.K = dict(zip(redemption_dates, K))
        self.barrier_rate = barrier_rate
        self.T = T
        self.F = F
        dates = []
        for i in range((redemption_dates[-1] - start_date).days + 1):
            dates.append(start_date + timedelta(days=i))
        self.dates = dates

    def get_price(self, initial_underlyings, mean, vols, corr, method, N, iter_num=None, seed=None, r=None, div=None,
                  exchange_rate=None):
        if method == "MC":
            s = GBM(s0=initial_underlyings, mean=mean, cov=vols * corr * vols.T, iter_num=iter_num, N=N, T=self.T,
                    div=div, seed=seed)
        if method == "FDM_3D":
            price = self.FDM_pricing(N=N, eval_date=self.start_date, vols=vols, corr=corr)
        return price

    def get_greek(self, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        greeks = {
            DELTA: self.get_delta(mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            GAMMA: self.get_gamma(mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            THETA: self.get_theta(mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                  exchange_rate=exchange_rate),
            VEGA: self.get_vega(mean=mean, cov=cov, iter_num=iter_num, N=N, seed=seed, r=r, div=div,
                                exchange_rate=exchange_rate)
        }
        return greeks

    def get_delta(self, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        pu = self.get_price(initial_underlyings=self.initial_underlyings * 1.01, mean=mean, vols=cov, iter_num=iter_num,
                            N=N, seed=seed, r=r, div=div, exchange_rate=exchange_rate)
        pd = self.get_price(initial_underlyings=self.initial_underlyings * 0.99, mean=mean, vols=cov, iter_num=iter_num,
                            N=N, seed=seed, r=r, div=div, exchange_rate=exchange_rate)
        return (pu - pd) / 0.02

    def get_gamma(self, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return

    def get_vega(self, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return

    def get_theta(self, mean, cov, iter_num, N, seed=None, r=None, div=None, exchange_rate=None):
        return

    def FDM_pricing(self, N, eval_date, vols, corr, IRTS, q):
        Nt = (self.dates[-1] - self.dates[0]).days
        dt = T / Nt
        u_NKI = np.zeros(tuple(N + 2))
        u_KI = deepcopy(u_NKI)
        step = self.initial_underlyings * 0.01
        h = step
        max_underlyings = step * (N + 1)
        min_underlyings = step * 0
        prices = np.array(
            [np.linspace(min_underlyings[0], max_underlyings[0], N[0] + 2),
             np.linspace(min_underlyings[1], max_underlyings[1], N[1] + 2),
             np.linspace(min_underlyings[2], max_underlyings[2], N[2] + 2),
             ])

        location_x = (np.abs(prices[0] - self.initial_underlyings[0])).argmin()
        location_y = (np.abs(prices[1] - self.initial_underlyings[1])).argmin()
        location_z = (np.abs(prices[2] - self.initial_underlyings[2])).argmin()

        # setting coefficient
        a = [[0] * N[0]] + [[0] * N[1]] + [[0] * N[2]]
        b = deepcopy(a)
        c = deepcopy(a)
        func = np.vectorize(get_coefficient)
        funcsteps = np.vectorize(func)
        for i in range(len(N)):
            a[i], b[i], c[i] = funcsteps(h=h[i], vols=vols[i], dt=dt, price=prices[i][1:N[i] + 1], r=IRTS[i], q=q[i])
            b[i][N[i] - 1] = b[i][N[i] - 1] + 2.0 * c[i][N[i] - 1]
            a[i][N[i] - 1] = a[i][N[i] - 1] - c[i][N[i] - 1]

        # Initial cond
        for x in range(N[0] + 2):
            for y in range(N[1] + 2):
                for z in range(N[2] + 2):
                    u_NKI = ELS_initial_cond(self.initial_underlyings, self.K, self.redemption_dates, self.coupon,
                                             self.F, self.barrier_rate, prices, u_NKI, x, y, z, KI=False)
                    u_KI = ELS_initial_cond(self.initial_underlyings, self.K, self.redemption_dates, self.coupon,
                                            self.F, self.barrier_rate, prices, u_KI, x, y, z, KI=True)

        # while iteration < Nt
        for _date in tqdm(list(reversed(self.dates))):
            u_NKI = get_payoff(u=u_NKI, redemption_dates=redemption_dates, coupon=coupon, F=F, _date=_date,
                               a=a, b=b, c=c, h=h, dt=dt, prices=prices, vols=vols, corr=corr, N=N)
            u_KI = get_payoff(u=u_KI, redemption_dates=redemption_dates, coupon=coupon, F=F, _date=_date,
                              a=a, b=b, c=c, h=h, dt=dt, prices=prices, vols=vols, corr=corr, N=N)

            # Early redemption
            if _date in redemption_dates:
                for x in range(N[0] + 2):
                    for y in range(N[1] + 2):
                        if min(prices[0][x] / self.initial_underlyings[0],
                               prices[1][y] / self.initial_underlyings[1],
                               prices[2][y] / self.initial_underlyings[2]) >= K[_date] * F:
                            u_NKI[x, y] = (1 + self.coupon[_date]) * F
                            u_KI[x, y] = (1 + self.coupon[_date]) * F

        return


if __name__ == '__main__':
    # example
    underlyings = [KS200, SX5E, SPX]
    initial_underlyings = np.array([301.25, 3300.16, 3258.44])
    start_date = datetime(2020, 7, 29)
    redemption_dates = np.array(
        [datetime(2021, 1, 28), datetime(2021, 7, 28), datetime(2022, 1, 27), datetime(2022, 7, 28),
         datetime(2023, 1, 26), datetime(2023, 7, 27)])
    coupon = [0.025, 0.05, 0.075, 0.1, 0.125, 0.15]
    K = [0.85, 0.85, 0.85, 0.80, 0.75, 0.65]
    barrier_rate = 0.65
    N = np.array([200, 200, 200])
    T = 3
    F = 1000

    self = ELS(underlyings=underlyings, initial_underlyings=initial_underlyings, start_date=start_date, F=F,
               coupon=coupon, redemption_dates=redemption_dates, K=K, barrier_rate=barrier_rate, T=T)

    # ELS pricing
    mean = np.array([0, 0, 0])
    vols = np.array([0.2417, 0.2703, 0.3013])
    corr = np.array([[1, 0.5504, 0.3432], [0.5504, 1, 0.7207], [0.3432, 0.7207, 1]])
    q = np.array([0, 0, 0])
    IRTS = np.array([0.0078, 0.005, 0.0058])
    ELS.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols, method="FDM_3D", N=N, )
    # target = 883.478
