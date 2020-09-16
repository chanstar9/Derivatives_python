# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 02. 06
"""

from datetime import datetime, timedelta
from tqdm import tqdm

from numpy import array
from itertools import combinations_with_replacement
import pandas as pd
from multiprocessing import Pool

from columns import *
from stock_process import *
from FDM.cal_func import *


class ELS:
    """
    To get ELS price, greek
    """

    def __init__(self, underlyings, initial_underlyings, start_date: datetime, F, redemption_dates: array, coupon: list,
                 K: list, barrier_rate: float, T: float, structure: str):
        for underlying in underlyings:
            if underlying not in INDICES:
                raise ValueError('{} is not registered.'.format(underlying))
        self.underlyings = underlyings
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
        if not structure:
            ValueError("Specify ELS Structure")
        self.structure = structure

    def get_price(self, initial_underlyings, mean, vols, corr, method, N=None, eval_date=None, iter_num=None, seed=None,
                  IRTS=None, div=None, exchange_rate=None):
        global price
        if "MC" in method:
            global underlying_scenario
            if method == "GBM_MC":
                underlying_scenario = GBM(initial_underlyings=initial_underlyings, mean=mean, vols=vols, corr=corr,
                                          iter_num=iter_num, N=len(self.dates), T=self.T, div=div, seed=seed)
            if method == "Levy_MC":
                # underlying_scenario = data_embedded_distribution()
                pass
            price = self.MC_pricing(underlying_scenario=underlying_scenario, IRTS=IRTS, iter_num=iter_num,
                                    eval_date=eval_date, structure=self.structure)

        if method == "FDM_3D":
            price = self.FDM_pricing(N=N, eval_date=eval_date, vols=vols, corr=corr, IRTS=IRTS, div=div)

        return price

    def get_greek(self, mean, vols, corr, iter_num, N, method, seed=None, r=None, div=None, exchange_rate=None):
        greeks = {
            DELTA: self.get_delta(mean=mean, vols=vols, corr=corr, iter_num=iter_num, N=N, method=method, seed=seed,
                                  r=r, div=div, exchange_rate=exchange_rate),
            GAMMA: self.get_gamma(mean=mean, vols=vols, corr=corr, iter_num=iter_num, N=N, method=method, seed=seed,
                                  r=r, div=div, exchange_rate=exchange_rate),
            THETA: self.get_theta(mean=mean, vols=vols, corr=corr, iter_num=iter_num, N=N, method=method, seed=seed,
                                  r=r, div=div, exchange_rate=exchange_rate),
            VEGA: self.get_vega(mean=mean, vols=vols, corr=corr, iter_num=iter_num, N=N, method=method, seed=seed, r=r,
                                div=div, exchange_rate=exchange_rate)
        }
        return greeks

    def get_delta(self, mean, vols, corr, iter_num, N, method, seed=None, r=None, div=None, exchange_rate=None):
        delta = []
        for i in range(len(self.underlyings)):
            if method == "GBM_MC":
                # monte carlo
                _arr = np.array([1, 1, 1])
                _arr[i] = _arr[i] * 1.01
                pu = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                _arr[i] = _arr[i] * 0.99 / 1.01
                pd = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                delta.append((pu - pd) / 0.02)
        return dict(zip(self.underlyings, delta))

    def get_gamma(self, mean, vols, corr, iter_num, N, method, seed=None, r=None, div=None, exchange_rate=None):
        # monte carlo의 경우 gamma 값이 이상함...
        gamma = []
        permu = list(combinations_with_replacement(self.underlyings, 2))
        for i in permu:
            if i[0] == i[1]:
                # plain gamma
                _location = self.underlyings.index(i[0])
                p = self.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols,
                                   corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                   div=div, exchange_rate=exchange_rate)
                _arr = np.array([1, 1, 1])
                _arr[_location] = _arr[_location] * 1.01
                pu = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                _arr[_location] = _arr[_location] * 0.99 / 1.01
                pd = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                gamma.append((pu - 2 * p + pd) / (0.01 ** 2))
            else:
                # cross gamma
                _location1 = self.underlyings.index(i[0])
                _location2 = self.underlyings.index(i[1])
                _arr = np.array([1, 1, 1])
                _arr[_location1] = _arr[_location1] * 1.01
                _arr[_location2] = _arr[_location2] * 1.01
                puu = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                     corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed,
                                     IRTS=r, div=div, exchange_rate=exchange_rate)
                _arr[_location2] = _arr[_location2] * 0.99 / 1.01
                pud = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                     corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed,
                                     IRTS=r, div=div, exchange_rate=exchange_rate)
                _arr[_location1] = _arr[_location1] * 0.99 / 1.01
                pdd = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                     corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed,
                                     IRTS=r, div=div, exchange_rate=exchange_rate)
                _arr[_location1] = _arr[_location1] * 1.01 / 0.99
                _arr[_location2] = _arr[_location2] * 1.01 / 0.99
                pdu = self.get_price(initial_underlyings=self.initial_underlyings * _arr, mean=mean, vols=vols,
                                     corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed,
                                     IRTS=r, div=div, exchange_rate=exchange_rate)
                gamma.append((puu - pud - pdu + pdd) / (0.01 ** 2))
        return dict(zip(permu, gamma))

    def get_vega(self, mean, vols, corr, iter_num, N, method, seed=None, r=None, div=None, exchange_rate=None):
        vega = []
        for i in range(len(self.underlyings)):
            if method == "GBM_MC":
                # monte carlo
                _arr = np.array([1, 1, 1])
                _arr[i] = _arr[i] * 1.0001
                pu = self.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols * _arr,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                _arr[i] = _arr[i] * 0.9999 / 1.0001
                pd = self.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols * _arr,
                                    corr=corr, method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r,
                                    div=div, exchange_rate=exchange_rate)
                vega.append((pu - pd) / 0.0002)
        vega = dict(zip(self.underlyings, vega))
        return vega

    def get_theta(self, mean, vols, corr, iter_num, N, method, seed=None, r=None, div=None, exchange_rate=None):
        # pu = self.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols, corr=corr,
        #                     method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r, div=div,
        #                     exchange_rate=exchange_rate)
        # pd = self.get_price(initial_underlyings=self.initial_underlyings, mean=mean, vols=vols, corr=corr,
        #                     method=method, N=N, eval_date=None, iter_num=iter_num, seed=seed, IRTS=r, div=div,
        #                     exchange_rate=exchange_rate)

        return

    def FDM_pricing(self, N, eval_date, vols, corr, IRTS, div):
        Nt = (self.dates[-1] - self.dates[0]).days
        dt = T / Nt
        u_NKI = np.zeros(tuple(N + 2))
        u_KI = deepcopy(u_NKI)
        h = self.initial_underlyings * 0.01
        max_underlyings = h * (N + 1)
        min_underlyings = h * 0
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
            a[i], b[i], c[i] = funcsteps(h=h[i], vols=vols[i], dt=dt, price=prices[i][1:N[i] + 1], r=IRTS[i], q=div[i])
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
            u_NKI = get_payoff(u=u_NKI, redemption_dates=self.redemption_dates, coupon=self.coupon, F=self.F,
                               _date=_date, a=a, b=b, c=c, h=h, dt=dt, prices=prices, vols=vols, corr=corr, N=N)
            u_KI = get_payoff(u=u_KI, redemption_dates=self.redemption_dates, coupon=self.coupon, F=self.F, _date=_date,
                              a=a, b=b, c=c, h=h, dt=dt, prices=prices, vols=vols, corr=corr, N=N)

            # Early redemption
            if _date in redemption_dates:
                for x in range(N[0] + 2):
                    for y in range(N[1] + 2):
                        for z in range(N[2] + 2):
                            if min(prices[0][x] / self.initial_underlyings[0],
                                   prices[1][y] / self.initial_underlyings[1],
                                   prices[2][z] / self.initial_underlyings[2]) >= self.K[_date] * F:
                                u_NKI[x, y, z] = (1 + self.coupon[_date]) * F
                                u_KI[x, y, z] = (1 + self.coupon[_date]) * F

            # adjust u_NKI
            u_NKI[:int(100 * barrier_rate), :, :] = deepcopy(u_KI[:int(100 * barrier_rate), :, :])
            u_NKI[int(100 * barrier_rate):, :int(100 * barrier_rate), :] = deepcopy(
                u_KI[int(100 * barrier_rate):, :int(100 * barrier_rate), :])
            u_NKI[int(100 * barrier_rate):, int(100 * barrier_rate):, :int(100 * barrier_rate)] = deepcopy(
                u_KI[int(100 * barrier_rate):, int(100 * barrier_rate):, :int(100 * barrier_rate)])
        return u_NKI, u_NKI[location_x - 1, location_y - 1, location_z - 1]

    def MC_pricing(self, underlying_scenario, IRTS, iter_num, eval_date, structure=None):
        prices_survives = np.ones((iter_num, 2))
        redem_loc = [idx for idx, value in enumerate(self.dates) if value in self.redemption_dates]
        if structure in ["Step Down KI", "Step Down NKI"]:
            for idx, redem_date in enumerate(self.redemption_dates):
                DF = np.exp(-IRTS[0] * (redem_date - eval_date).days / 365)
                redem_occur = (underlying_scenario[:, redem_loc[idx], :] / initial_underlyings).min(axis=1) >= self.K[
                    redem_date]
                survive_redem = np.array(prices_survives[:, 1] * redem_occur, dtype=bool)
                prices_survives[:, 1] = prices_survives[:, 1] * ~redem_occur
                if redem_date == self.redemption_dates[-1]:
                    survive_not_redem = np.array(prices_survives[:, 1], dtype=bool)
                    if structure == "Step Down KI":
                        # criteria for KI or NKI
                        prices_survives[survive_redem, 0] = np.apply_along_axis(
                            lambda x: DF * (1 + self.coupon[redem_date]) * self.F if x else
                            (underlying_scenario[survive_redem, -1, :] / initial_underlyings).min(axis=1) * self.F,
                            0, (underlying_scenario[survive_redem, :, :] / initial_underlyings).min(axis=2).min(
                                axis=1) >= self.barrier_rate)
                    else:
                        prices_survives[survive_redem, 0] = DF * (1 + self.coupon[redem_date]) * self.F
                    prices_survives[survive_not_redem, 0] = (underlying_scenario[survive_not_redem, -1,
                                                             :] / initial_underlyings).min(axis=1) * self.F * DF
                else:
                    prices_survives[survive_redem, 0] = DF * (1 + self.coupon[redem_date]) * self.F
        if structure in ["Step Down KI - Mtly cpn", "Step Down NKI - Mtly cpn"]:
            pass
        if structure in ["Step Down KI - Lizard", "Step Down NKI - Lizard"]:
            pass
        return prices_survives[:, 0].mean()


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
    T = 3
    F = 1000
    structure = "Step Down NKI"

    self = ELS(underlyings=underlyings, initial_underlyings=initial_underlyings, start_date=start_date, F=F,
               coupon=coupon, redemption_dates=redemption_dates, K=K, barrier_rate=barrier_rate, T=T,
               structure=structure)

    # ELS pricing
    N = np.array([200, 200, 200])
    mean = np.array([0, 0, 0])
    vols = np.array([0.35, 0.4, 0.45])
    corr = np.array([[1, 0.5504, 0.3432], [0.5504, 1, 0.7207], [0.3432, 0.7207, 1]])
    div = np.array([0.023, 0.0256, 0.0432])
    IRTS = np.array([0.014, 0.01, 0.01])

    # FDM
    # fdm, p = self.get_price(mean=mean, vols=vols, corr=corr, method="FDM_3D", N=N, IRTS=IRTS, div=div)
    # print(p)

    # monte carlo
    price = self.get_price(initial_underlyings=initial_underlyings, mean=mean, vols=vols, corr=corr, method="GBM_MC",
                           eval_date=start_date, iter_num=10000, IRTS=IRTS, div=div)
    print(price)
    # target = 883.478
