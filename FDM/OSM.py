# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 10 .02
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm import tqdm


def TDMAsolver(aa, bb, cc, dd):
    nf = len(dd[0])  # number of edivuations
    acc, bcc, ccc, dcc = map(np.array, (aa, bb, cc, dd))  # copy the array
    for it in range(1, nf):
        mc = acc[it - 1] / bcc[it - 1]
        bcc[it] = bcc[it] - mc * ccc[it - 1]
        dcc[:, it] = dcc[:, it] - mc * dcc[:, it - 1]

    xcc = dcc
    xcc[:, -1] = dcc[:, -1] / bcc[-1]

    for il in reversed(range(0, nf - 1)):
        xcc[:, il] = (dcc[:, il] - ccc[il] * xcc[:, il + 1]) / bcc[il]

    return xcc


def get_d(u, x, y, corr):
    return (1 / 2) * corr * vols[0] * vols[1] * price[0][x] * price[1][y] * (
            u[x + 1, y + 1] - u[x + 1, y - 1] - u[x - 1, y + 1] + u[x - 1, y - 1]) / (4 * h * k) + u[x, y] / dt


def ELS_initial_cond(underlying_prices, u, x, y, KI):
    if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= K[redemption_dates[-1]]:
        u[x, y] = (1 + coup[redemption_dates[-1]]) * F
    else:
        if not KI:  # If a knock-in does not occur before maturity
            # If a knock-in does not occur at maturity
            if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= barrier_rate:
                u[x, y] = (1 + coup[redemption_dates[-1]]) * F
            else:  # If a knock-in does occur at maturity
                u[x, y] = min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) * F
        else:
            u[x, y] = min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) * F
    return u


def get_coefficient(vols, price, r, q):
    b = -(vols * price) ** 2 / (2 * (h ** 2))
    c = -(vols * price) ** 2 / (2 * (h ** 2)) - (r - q) * price / h
    a = 1 / dt + (vols * price) ** 2 / (h ** 2) + (r - q) * price / h + r / 2
    return a, b, c


def get_payoff(u, date_, a, b, c, corr):
    # x - direction
    d = np.zeros((N[0], N[1])).T
    for y in range(1, N[1] + 1):
        d[y - 1] = np.fromiter((get_d(u, x, y, corr) for x in range(1, N[0] + 1)), dtype=float)
    u[1:N[0] + 1, 1:-1] = TDMAsolver(b[0][1:], a[0], c[0][:-1], d)

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coup[(redemption_dates[redemption_dates >= date_]).min()]) * F

    # y - direction
    d = np.zeros((N[0], N[1]))
    for x in range(1, N[0] + 1):
        d[x - 1] = np.fromiter((get_d(u, x, y, corr) for y in range(1, N[1] + 1)), dtype=float)
    u[1:-1, 1:N[1] + 1] = TDMAsolver(b[1][1:], a[1][:], c[1][:-1], d)

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coup[(redemption_dates[redemption_dates >= date_]).min()]) * F
    return u


def OSM_pricing(eval_date, underlying_prices, vols, corr, plot=False):
    # set computational domain
    u_NKI = np.zeros(tuple(np.array(N) + 2))
    u_KI = deepcopy(u_NKI)

    # Coefficient of x
    a = [[0] * N[0]] + [[0] * N[1]]
    b = deepcopy(a)
    c = deepcopy(a)

    func = np.vectorize(get_coefficient)
    funcsteps = np.vectorize(func)

    for i in range(len(N)):
        a[i], b[i], c[i] = funcsteps(vols[i], price[i][1:N[i] + 1], r[i], q[i])
        a[i][0] = a[i][0] + 2.0 * b[i][0]
        c[i][0] = c[i][0] - b[i][0]
        b[i][N[i] - 1] = b[i][N[i] - 1] - c[i][N[i] - 1]
        a[i][N[i] - 1] = a[i][N[i] - 1] + 2.0 * c[i][N[i] - 1]

    # Initial condition
    for x in range(N[0] + 2):
        for y in range(N[1] + 2):
            # If redemption condition is satisfied
            u_NKI = ELS_initial_cond(underlying_prices, u_NKI, x, y, KI=False)
            u_KI = ELS_initial_cond(underlying_prices, u_KI, x, y, KI=True)

    # while iteration < Nt :
    for date_ in tqdm(list(reversed(dates[dates.index(eval_date):-1]))):
        # with Pool(2) as p:
        #     results = [p.apply_async(get_payoff, t) for t in
        #                [[u_NKI, a, b, c, d, iteration], [u_KI, a, b, c, d, iteration]]]
        #     for result in results:
        #         result.wait()
        #     u_NKI, u_KI = [result.get() for result in results]
        #     p.close()
        #     p.join()
        u_NKI, u_KI = get_payoff(u_NKI, date_, a, b, c, corr), get_payoff(u_KI, date_, a, b, c, corr)

        # Early redemption
        if date_ in redemption_dates:
            for x in range(N[0] + 2):
                for y in range(N[1] + 2):
                    if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= K[date_] * F:
                        u_NKI[x, y] = (1 + coup[date_]) * F
                        u_KI[x, y] = (1 + coup[date_]) * F

        # adjust u_NKI
        u_NKI[:int(100 * barrier_rate), :] = deepcopy(u_KI[:int(100 * barrier_rate), :])
        u_NKI[int(100 * barrier_rate):, :int(100 * barrier_rate)] = deepcopy(
            u_KI[int(100 * barrier_rate):, :int(100 * barrier_rate)])

    fdm_price = u_NKI[location_x - 1, location_y - 1]

    if plot:
        # graph of ELS price
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xnew, ynew = np.meshgrid(price[1], price[0])
        surface = ax.plot_surface(xnew, ynew, u_NKI.T, cmap=cm.winter)
        ax.set_xlabel('S&P500')
        ax.set_ylabel('EUROSTOXX50')
        ax.set_zlabel('price')
        plt.title('ELS_surface')
        fig.savefig('picture/els.png')
        plt.show()

    return u_NKI, fdm_price


def get_delta(Total_ELSPrice):
    delta_x = (Total_ELSPrice[1:, location_y] - Total_ELSPrice[:-1, location_y]) / h
    delta_y = (Total_ELSPrice[location_x, 1:] - Total_ELSPrice[location_x, :-1]) / k
    return delta_x, delta_y, delta_x[location_x], delta_y[location_y]


def get_gamma(Total_ELSPrice):
    gamma_xx = (Total_ELSPrice[2:, location_y] - 2 * Total_ELSPrice[1:-1, location_y] +
                Total_ELSPrice[:-2, location_y]) / (pow(h, 2))
    gamma_yy = (Total_ELSPrice[location_x, 2:] - 2 * Total_ELSPrice[location_x, 1:-1] +
                Total_ELSPrice[location_x, :-2]) / (pow(h, 2))
    gamma_xy = (Total_ELSPrice[2:, 2:] + Total_ELSPrice[:-2, :-2] - Total_ELSPrice[2:, :-2] -
                Total_ELSPrice[:-2, 2:]) / (h * k)
    return gamma_xx, gamma_xy, gamma_yy, gamma_xx[location_x], gamma_yy[location_y], gamma_xy[location_x, location_y]


def get_vega(eval_date, Total_ELSPrice, revised_vols):
    vega_x = (Total_ELSPrice - OSM_pricing(eval_date, underlying_prices, [revised_vols[0], vols[1]], corr))
    vega_y = (Total_ELSPrice - OSM_pricing(eval_date, underlying_prices, [vols[0], revised_vols[1]], corr))
    return vega_x, vega_y


def get_rho(eval_date, Total_ELSPrice, revised_corr):
    rho = (Total_ELSPrice - OSM_pricing(eval_date, underlying_prices, vols, revised_corr))
    return rho


if __name__ == '__main__':
    # date
    start = datetime(2019, 9, 6)
    end = datetime(2022, 9, 2)
    dates = []
    delta_save = []
    for i in range((end - start).days + 1):
        dates.append(start + timedelta(days=i))
    redemption_dates = np.array(
        [datetime(2020, 3, 5), datetime(2020, 9, 4), datetime(2021, 3, 5), datetime(2021, 9, 3),
         datetime(2022, 3, 4), datetime(2022, 9, 2)])

    stop_date = datetime(2019, 10, 17)
    hedge_dates = []
    for i in range((stop_date - start).days + 1):
        hedge_dates.append(start + timedelta(days=i))
    T = 3
    Nt = (end - start).days
    dt = T / Nt

    # property
    F = 10000  # face value
    coup = dict(zip(redemption_dates,
                    [0.018, 0.036, 0.054, 0.072, 0.09, 0.108]))  # Rate of return on each early redemption date
    K = dict(
        zip(redemption_dates, [0.9, 0.9, 0.9, 0.85, 0.8, 0.75]))  # Exercise price on each early redemption date
    barrier_rate = 0.5

    N = np.array([200, 200])
    # hedge_info = pd.read_csv('data/total_els_data.csv', parse_dates=True, index_col=0)
    underlying = pd.read_csv('data/total_hist_index_data.csv', parse_dates=True, index_col=0)
    interest = pd.read_csv('data/interest_rate/total_interest_data.csv', parse_dates=True, index_col=0)
    # corr_info = pd.read_csv('data/final_corr_vol.csv', parse_dates=True, index_col=0)
    ret = ((underlying - underlying.shift(1)) / underlying.shift(1)).dropna()

    delta_xx = []
    delta_yy = []
    bond = []
    future_position_snp = []
    future_position_euro = []
    delta_hedge = []
    pp = []

    for eval_date in hedge_dates:
        # price
        underlying_prices = underlying.loc[eval_date].values  # 19.09.06 prices of
        vols = np.array([0.1570, 0.1465])  # volatility of underlying: average of 6 months and 3 years
        corr = 0.62120  # correlation between  underlying1 and underlying2: average of 6 months and 3 years
        step = underlying_prices * 0.01
        h, k = step[0], step[1]
        max_prices = step * (N + 1)  # max prices of underlying
        min_prices = step * 0  # min prices of underlying

        price = np.array(
            [np.linspace(min_prices[0], max_prices[0], N[0] + 2),
             np.linspace(min_prices[1], max_prices[1], N[1] + 2)])
        location_x = (np.abs(price[0] - underlying_prices[0])).argmin()
        location_y = (np.abs(price[1] - underlying_prices[1])).argmin()
        # interest & dividend
        rd = interest.loc[eval_date].values[2] * 0.01
        rf = interest.loc[eval_date].values[0:2] * 0.01
        r = rd - rf
        q = np.array([0.0185, 0.0351])  # dividend rate of the underlying

        Total_ELSPrice, fdm_price = OSM_pricing(eval_date, underlying_prices, vols, corr, plot=False)

        pp.append(fdm_price)
        # delta_xx.append((get_delta(Total_ELSPrice)[2] * hedge_info[['USD_KRW']].loc[eval_date].values / 10)[-1])
        # delta_yy.append((get_delta(Total_ELSPrice)[3] * hedge_info[['EURO_KRW']].loc[eval_date].values / 250)[-1])
        #
        # if eval_date == start:
        #     print('test')
        #     _bond = fdm_price - delta_xx * hedge_info[['S&P500FUTURES']].loc[eval_date].values \
        #             - delta_yy * hedge_info[['EUROFUTURES']].loc[eval_date].values
        #     bond.append(_bond[-1])
        #     future_position_snp.append(delta_xx[-1] * hedge_info[['S&P500FUTURES']].loc[eval_date].values[-1])
        #     future_position_euro.append(delta_yy[-1] * hedge_info[['EUROFUTURES']].loc[eval_date].values[-1])
        #     delta_hedge.append(fdm_price)
        #
        # else:
        #     future_position_snp.append((delta_xx[(eval_date - start).days] - delta_xx[(eval_date - start).days - 1])
        #                                * hedge_info[['S&P500FUTURES']].loc[eval_date].values[0])
        #     future_position_euro.append(
        #         (delta_yy[(eval_date - start).days] - delta_yy[(eval_date - start).days - 1])
        #         * hedge_info[['EUROFUTURES']].loc[eval_date].values[0])
        #
        #     _bond = bond[(eval_date - start).days - 1] * np.exp(rd / 365) - future_position_snp[
        #         (eval_date - start).days] \
        #             - future_position_euro[(eval_date - start).days] + future_position_snp[
        #                 (eval_date - start).days - 1] \
        #             + future_position_euro[(eval_date - start).days - 1]
        #     bond.append(_bond)
        #
        #     delta_hedge.append(future_position_snp[(eval_date - start).days] + future_position_euro[
        #         (eval_date - start).days] + _bond)