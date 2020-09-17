# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 
"""
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from multiprocessing import Pool
from tqdm import tqdm

# parameters of underlying
underlying_prices = np.array([2976, 3484.70])  # 19.09.06 prices of underlying
vols = np.array([0.1570, 0.1465])  # volatility of underlying: average of 6 months and 3 years
corr = 0.62120  # correlation between  underlying1 and underlying2: average of 6 months and 3 years
q = np.array([0.0185, 0.0351])  # dividend rate of the underlying
mu = np.array([0.01378, -0.05304])  # IRS
# mu = np.array([0.01444, -0.0053])   # FRA
# mu = np.array([0.013545, -0.00828]) # 3-years treasury bill
# parameters of ELS
F = 10000  # face value
T = 3  # maturity
coup = 0.018 * np.array(range(1, T * 2 + 1))  # Rate of return on each early redemption date
K = np.array([0.9, 0.9, 0.9, 0.85, 0.8, 0.75])  # Exercise price on each early redemption date
barrier_rate = 0.5

# parameters for testing
rd = 0.0151
rf = np.array([0.01378, -0.05304])
r = rd - rf
redemption_term = 100
Nt = 6 * redemption_term
N = np.array([200, 200])
dt = T / Nt
step = underlying_prices * 0.01
h, k = step[0], step[1]
max_prices = step * (N + 1)  # max prices of underlying
min_prices = step * 0  # min prices of underlying
price = np.array(
    [np.linspace(min_prices[0], max_prices[0], N[0] + 2), np.linspace(min_prices[1], max_prices[1], N[1] + 2)])
location_x = (np.abs(price[0] - underlying_prices[0])).argmin()
location_y = (np.abs(price[1] - underlying_prices[1])).argmin()


def TDMAsolver(aa, bb, cc, dd):
    nf = len(dd)  # number of edivuations
    ac, bc, cc, dc = map(np.array, (aa, bb, cc, dd))  # copy the array
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def get_d(u, x, y, corr):
    return (1 / 2) * corr * vols[0] * vols[1] * price[0][x] * price[1][y] * (
            u[x + 1, y + 1] - u[x + 1, y - 1] - u[x - 1, y + 1] + u[x - 1, y - 1]) / (4 * h * k) + u[x, y] / dt


def ELS_initial_cond(underlying_prices, u, x, y, KI):
    if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= K[5]:
        u[x, y] = (1 + coup[5]) * F
    else:
        if not KI:  # If a knock-in does not occur before maturity
            # If a knock-in does not occur at maturity
            if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= barrier_rate:
                u[x, y] = (1 + coup[5]) * F
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


def get_payoff(u, a, b, c, corr, iteration):
    # x - direction
    d = np.zeros((N[0], N[1]))
    for y in range(1, N[1] + 1):
        d[0] = np.fromiter((get_d(u, x, y, corr) for x in range(1, N[0] + 1)), dtype=float)
        u[1:N[0] + 1, y] = TDMAsolver(b[0][1:], a[0], c[0][:-1], d[0])

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coup[5 - int(np.floor(iteration / redemption_term))]) * F

    # y - direction
    for x in range(1, N[0] + 1):
        d[1] = np.fromiter((get_d(u, x, y, corr) for y in range(1, N[1] + 1)), dtype=float)
        u[x, 1:N[1] + 1] = TDMAsolver(b[1][1:], a[1][:], c[1][:-1], d[1])

    # Linear boundary condition
    u[0, 1:N[1] + 1] = 2 * u[1, 1:N[1] + 1] - u[2, 1:N[1] + 1]
    u[N[0] + 1, 1:N[1] + 1] = 2 * u[N[0], 1:N[1] + 1] - u[N[0] - 1, 1:N[1] + 1]
    u[0:N[0] + 2, 0] = 2 * u[0:N[0] + 2, 1] - u[0:N[0] + 2, 2]
    u[0:N[0] + 2, N[1] + 1] = 2 * u[0:N[0] + 2, N[1]] - u[0:N[0] + 2, N[1] - 1]
    u[0, 0] = 0
    u[N[0] + 1, N[1] + 1] = (1 + coup[5 - int(np.floor(iteration / redemption_term))]) * F

    return u


def OSM_pricing(underlying_prices, vols, corr, plot=False):
    # set computational domain
    u_NKI = np.zeros(tuple(np.array(N) + 2))
    u_KI = deepcopy(u_NKI)

    # Coefficient of x
    a = [[0] * N[0]] + [[0] * N[1]]
    b = deepcopy(a)
    c = deepcopy(a)
    # d = deepcopy(a)

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

    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xnew, ynew = np.meshgrid(price[0], price[1])
        surface = ax.plot_surface(xnew, ynew, u_NKI, cmap=cm.winter)
        ax.set_xlabel('S&P500')
        ax.set_ylabel('EUROSTOXX50')
        ax.set_zlabel('NKI Initial Condition')
        fig.savefig('picture/nki.png')
        plt.show()

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xnew, ynew = np.meshgrid(price[0], price[1])
        surface = ax.plot_surface(xnew, ynew, u_KI, cmap=cm.winter)
        ax.set_xlabel('S&P500')
        ax.set_ylabel('EUROSTOXX50')
        ax.set_zlabel('KI Initial Condition')
        fig.savefig('picture/ki.png')
        plt.show()

    iteration = 0
    # while iteration < Nt :
    for _ in tqdm(range(Nt)):
        # with Pool(2) as p:
        #     results = [p.apply_async(get_payoff, t) for t in
        #                [[u_NKI, a, b, c, d, iteration], [u_KI, a, b, c, d, iteration]]]
        #     for result in results:
        #         result.wait()
        #     u_NKI, u_KI = [result.get() for result in results]
        #     p.close()
        #     p.join()
        u_NKI, u_KI = get_payoff(u_NKI, a, b, c, corr, iteration), get_payoff(u_KI, a, b, c, corr, iteration)

        # Early redemption
        if np.mod(iteration, redemption_term) == 0 and iteration < Nt and iteration != 0:
            for x in range(N[0] + 2):
                for y in range(N[1] + 2):
                    if min(price[0][x] / underlying_prices[0], price[1][y] / underlying_prices[1]) >= K[
                        5 - int(np.floor(iteration / redemption_term))] * F:
                        u_NKI[x, y] = (1 + coup[5 - int(np.floor(iteration / redemption_term))]) * F
                        u_KI[x, y] = (1 + coup[5 - int(np.floor(iteration / redemption_term))]) * F

        # adjust u_NKI
        u_NKI[:int(100 * barrier_rate), :] = deepcopy(u_KI[:int(100 * barrier_rate), :])
        u_NKI[int(100 * barrier_rate):, :int(100 * barrier_rate)] = deepcopy(
            u_KI[int(100 * barrier_rate):, :int(100 * barrier_rate)])

        # next step
        iteration += 1

    fdm_price = u_NKI[location_x - 1, location_y - 1]

    if plot:
        # graph of ELS price
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        xnew, ynew = np.meshgrid(price[1], price[0])
        surface = ax.plot_surface(xnew, ynew, u_NKI.T, cmap=cm.winter)
        ax.set_xlabel('S&P500')
        ax.set_ylabel('EUROSTOXX50')
        ax.set_zlabel('ELS')
        plt.title('ELS_surface')
        fig.savefig('picture/els.png')
        plt.show()

    return u_NKI, fdm_price


def delta(Total_ELSPrice):
    delta_x = (Total_ELSPrice[1:, location_y] - Total_ELSPrice[:-1, location_y]) / h
    delta_y = (Total_ELSPrice[location_x, 1:] - Total_ELSPrice[location_x, :-1]) / k

    fig1 = plt.figure(1)
    plt.plot(price[0][:-1], delta_x)
    plt.plot(underlying_prices[0] * barrier_rate * np.ones(len(delta_x)), delta_x)
    plt.xlabel('S&P500')
    plt.ylabel('Delta')
    plt.title('Delta Graph')
    fig1.savefig('picture/delta_x.png')
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(price[1][:-1], delta_y)
    plt.plot(underlying_prices[1] * barrier_rate * np.ones(len(delta_y)), delta_y)
    plt.xlabel('EUROSTOXX50')
    plt.ylabel('Delta')
    plt.title('Delta Graph')
    fig2.savefig('picture/delta_y.png')
    plt.show()

    return delta_x[location_x], delta_y[location_y]


def gamma(Total_ELSPrice):
    gamma_xx = (Total_ELSPrice[2:, location_y] - 2 * Total_ELSPrice[1:-1, location_y] + Total_ELSPrice[:-2,
                                                                                        location_y]) / (pow(h, 2))
    gamma_yy = (Total_ELSPrice[location_x, 2:] - 2 * Total_ELSPrice[location_x, 1:-1]
                + Total_ELSPrice[location_x, :-2]) / (pow(h, 2))
    gamma_xy = (Total_ELSPrice[2:, 2:] + Total_ELSPrice[:-2, :-2]
                - Total_ELSPrice[2:, :-2] - Total_ELSPrice[:-2, 2:]) / (h * k)
    fig1 = plt.figure(1)
    plt.plot(price[0][:-2], gamma_xx)
    plt.xlabel('S&P500')
    plt.ylabel('Gamma_xx')
    plt.title('Gamma Graph')
    fig1.savefig('picture/gamma_xx.png')
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(price[1][:-2], gamma_yy)
    plt.xlabel('EUROSTOXX50')
    plt.ylabel('Gamma_yy')
    plt.title('Gamma Graph')
    fig2.savefig('picture/gamma_yy.png')
    plt.show()

    fig3 = plt.figure(3)
    ax = fig3.gca(projection='3d')
    xnew, ynew = np.meshgrid(price[0][1:-1], price[1][1:-1])
    surface = ax.plot_surface(xnew, ynew, gamma_xy.T, cmap=cm.coolwarm)
    cset = ax.contourf(xnew, ynew, gamma_xy.T, zdir='z', offset=-1, cmap=cm.ocean)
    cset = ax.contourf(xnew, ynew, gamma_xy.T, zdir='x', offset=-1, cmap=cm.ocean)
    cset = ax.contourf(xnew, ynew, gamma_xy.T, zdir='y', offset=-1, cmap=cm.ocean)
    fig3.colorbar(surface, shrink=0.5, aspect=5)
    ax.set_xlabel('S&P500')
    ax.set_ylabel('EUROSTOXX50')
    ax.set_zlabel('Gamma_xy')
    plt.title('Gamma Graph')
    fig3.savefig('picture/gamma_xy.png')
    plt.show()

    return gamma_xx[location_x], gamma_yy[location_y], gamma_xy[location_x, location_y]


def vega(fdm_price):
    max_underlying = np.round(2 * underlying_prices)
    max_underlyingP = np.linspace(1, max_underlying, 10)
    vega1_list = []
    vega2_list = []

    # for i in (range(len(max_underlyingP))):
    #     new_underlying = max_underlyingP[i]
    #     vega1_els = OSM_pricing(new_underlying, vols, corr)[1] - fdm_price
    #     vega1_list.append(vega1_els)
    #
    # for j in (range(len(max_underlyingP))):
    #     vega2_els = OSM_pricing(max_underlyingP[j], vols, corr)[1] - fdm_price
    #     vega2_list.append(vega2_els)

    with Pool(10) as p:
        results = [p.apply_async(OSM_pricing, (underlying_prices_, vols, corr))
                   for underlying_prices_ in max_underlyingP]
        for r in results:
            r.wait()
        results = np.array([result.get()[1] for result in results])
        vega1_list = (results - fdm_price) / (max_underlyingP[:, 0] - underlying_prices[0])
        a = pd.DataFrame(vega1_list)
        a.to_csv('data/vega_x.csv')
        p.close()
        p.join()

    with Pool(10) as p:
        results = [p.apply_async(OSM_pricing, (underlying_prices_, vols, corr))
                   for underlying_prices_ in max_underlyingP]
        for r in results:
            r.wait()
        results = np.array([result.get()[1] for result in results])
        vega2_list = (results - fdm_price) / (max_underlyingP[:, 1] - underlying_prices[1])
        a = pd.DataFrame(vega2_list)
        a.to_csv('data/vega_y.csv')
        p.close()
        p.join()

    fig1 = plt.figure(1)
    plt.plot(max_underlyingP[:, 0], vega1_list)
    plt.xlabel('Change in S&P500 Volatility ')
    plt.ylabel('Vega_x')
    plt.title('Vega Graph with fixed volatility of EUROSTOXX50')
    fig1.savefig('picture/vega_x.png')
    plt.show()

    fig2 = plt.figure(2)
    plt.plot(max_underlyingP[:, 1], vega2_list)
    plt.xlabel('Change in EUROSTOXX50 Volatility')
    plt.ylabel('Vega_y')
    plt.title('Vega Graph with fixed volatility of S&P500')
    fig2.savefig('picture/vega_y.png')
    plt.show()
    return vega1_list, vega2_list


def rho_sensitivity(fdm_price):
    with Pool(21) as p:
        results = [p.apply_async(OSM_pricing, (underlying_prices, vols, corr_))
                   for corr_ in 0.1 * np.array(range(-10, 11))]
        for r in results:
            r.wait()
        results = np.array([result.get()[1] for result in results])

    # for i in tqdm(range(-10, 11)):
    #     rho = i * 0.1
    #     rho_els = OSM_pricing(underlying_prices, vols, rho)[1] - fdm_price
    #     rho_v = rho_els / (rho - corr)
    #     rho_els_list.append(rho_v)
    rho_els_list = (results - fdm_price) / (0.1 * np.array(range(-10, 11)) - corr)
    a = pd.DataFrame(rho_els_list)
    a.to_csv('data/rho_sensitivity.csv')

    fig1 = plt.figure(1)
    plt.plot([0.1 * i - corr for i in range(-10, 11)], rho_els_list)
    plt.xlabel('Change in correlation')
    plt.ylabel('rho_sensitivity')
    plt.title('rho sensitivity Graph')
    fig1.savefig('picture/rho.png')
    plt.show()


if __name__ == '__main__':
    Total_ELSPrice, fdm_price = OSM_pricing(underlying_prices, vols, corr, plot=True)
    # delta(Total_ELSPrice)
    # gamma(Total_ELSPrice)
    # vega(fdm_price)
    rho_sensitivity(fdm_price)
    print(fdm_price)