import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate as interpolate
import math as mth
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages


# Black Scholes formula
def BlackScholes(type, S0, K, r, sigma, T, q):
    def d1(S0, K, r, sigma, T, q):
        return (np.log(S0 / K) + (r - q + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    def d2(S0, K, r, sigma, T, q):
        return (np.log(S0 / K) + (r - q - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))

    if type == "C":
        return S0 * np.exp(- q * T) * norm.cdf(d1(S0, K, r, sigma, T, q)) - K * np.exp(-r * T) * norm.cdf(
            d2(S0, K, r, sigma, T, q))
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2(S0, K, r, sigma, T, q)) - S0 * np.exp(-q * T) * norm.cdf(
            -d1(S0, K, r, sigma, T, q))


# Add column LogStrike to data
def logstrike(K, T, S0, r, q): return np.log(K / S0 * np.exp(-(r - q) * T))


# Helper function to solve for implied volatility
def bsaux(sigma, type, S0, K, r, T, q, C): return BlackScholes(type, S0, K, r, sigma, T, q) - C


# Raw SVI parametrization
def rawSVI(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# Hyperbola asymptotes parametrisation
def straightSVI(x, m1, m2, q1, q2, c):
    return ((m1 + m2) * x + q1 + q2 + np.sqrt(
        ((m1 + m2) * x + q1 + q2) ** 2 - 4 * (m1 * m2 * x ** 2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))) / 2


def straightSVIp(x, m1, m2, q1, q2, c):
    H = np.sqrt(((m1 + m2) * x + q1 + q2) ** 2 - 4 * (m1 * m2 * x ** 2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))
    return ((m1 + m2) + ((m1 + m2) * ((m1 + m2) * x + q1 + q2) - 4 * m1 * m2 * x - 2 * (m1 * q2 + m2 * q1)) / H) / 2


def straightSVIpp(x, m1, m2, q1, q2, c):
    H = np.sqrt(((m1 + m2) * x + q1 + q2) ** 2 - 4 * (m1 * m2 * x ** 2 + (m1 * q2 + m2 * q1) * x + q1 * q2 - c))
    A = (2 * (m1 + m2) ** 2 - 8 * m1 * m2) / H
    B = (2 * (m1 + m2) * ((m1 + m2) * x + q1 + q2) - 8 * m1 * m2 * x - 4 * (m1 * q2 + m2 * q1)) ** 2 / H ** 3 / 2
    return (A - B) / 4


# Alternative parametrisation including the parabola
def stdSVI(x, a0, a1, a2, a3, a4):
    return (-a1 * x - a3 + np.sqrt((a1 * x + a3) ** 2 - 4 * a0 * (a2 * x ** 2 + x + a4))) / (2 * a0)


# Obtain asymptotic parameters from alternative parametrization
def std2straight(a):
    m1 = -a[1] / a[0] / 2. - np.sqrt((a[1] / a[0]) ** 2 / 4. - a[2] / a[0])
    m2 = -a[1] / a[0] / 2. + np.sqrt((a[1] / a[0]) ** 2 / 4. - a[2] / a[0])
    q1 = (1 + m1 * a[3]) / a[0] / (m2 - m1)
    q2 = (1 + m2 * a[3]) / a[0] / (m1 - m2)
    c = q1 * q2 - a[4] / a[0]
    return [m1, m2, q1, q2, c]


# Obtain rawSVI parameters from asymptotic parametrization
def straight2raw(chi):
    a = (chi[0] * chi[3] - chi[1] * chi[2]) / (chi[0] - chi[1])
    b = abs(chi[0] - chi[1]) / 2.
    rho = (chi[0] + chi[1]) / abs(chi[0] - chi[1])
    m = -(chi[2] - chi[3]) / (chi[0] - chi[1])
    sigma = np.sqrt(4 * chi[4]) / abs(chi[0] - chi[1])
    return [a, b, rho, m, sigma]


# Calculate risk neutral density wrt logstrike
def RND(k, m1, m2, q1, q2, c):
    w = straightSVI(k, m1, m2, q1, q2, c)
    wp = straightSVIp(k, m1, m2, q1, q2, c)
    wpp = straightSVIpp(k, m1, m2, q1, q2, c)
    g = (1. - k * wp / (2. * w)) ** 2 - wp ** 2 / 4. * (1. / w + 1. / 4.) + wpp / 2.
    return g / np.sqrt(2 * np.pi * w) * np.exp(-0.5 * ((-k - w / 2.) ** 2 / w))


# Do some plots as pdf files
def doplots(basefn, expirs, data, chi, grid, S0, r, q):
    i = 0
    for T in expirs:
        i += 1
        pp = PdfPages(basefn + '-' + str(i) + '.pdf')
        plt.figure(i, figsize=(8.0, 10.0))
        plt.subplot(311)
        plt.title('Slice ' + str(i) + ': Expiration T=' + str(T))
        plt.xlabel('Log-Strike')
        plt.ylabel('Option price')
        t = data.loc[data['Expiration'] == T, 'LogStrike']
        bid = data.loc[data['Expiration'] == T, 'Bid_price']
        ask = data.loc[data['Expiration'] == T, 'Ask_price']
        tt = grid[T]
        w = [straightSVI(k, chi.loc[T, 'm1'], chi.loc[T, 'm2'], chi.loc[T, 'q1'], chi.loc[T, 'q2'], chi.loc[T, 'c']) for
             k in grid[T]]
        m = [BlackScholes("C", S0, K, r, sig, T, q) for K, sig in zip(grid.index, np.sqrt(np.array(w) / T))]
        plt.plot(t, bid, 'bx', tt, m, 'k', markersize=3)
        plt.plot(t, ask, 'rx', tt, m, 'k', markersize=3)
        plt.subplot(312)
        plt.xlabel('Log-Strike')
        plt.ylabel('Implied volatility')
        iv = data.loc[data['Expiration'] == T, 'IV']
        cv = np.sqrt(np.array(w) / T)
        plt.plot(t, iv, 'bo', tt, cv, 'k', markersize=3)
        plt.subplot(313)
        plt.xlabel('Log-Strike')
        plt.ylabel('Risk neutral density')
        p = [RND(k, chi.loc[T, 'm1'], chi.loc[T, 'm2'], chi.loc[T, 'q1'], chi.loc[T, 'q2'], chi.loc[T, 'c']) for k in
             grid[T]]
        plt.plot(tt, p, 'k')
        plt.savefig(pp, format='pdf')
        pp.close()
