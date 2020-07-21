# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 09. 27
"""

import random
import numpy as np
from scipy.stats import norm


def GBM(s0, mean, cov, iter_num, N, T, div=None, seed=None):
    """
    Geometric brownian motion
    :return: axis0=iter_num, axis1=time, axis2= # of underlying
    """
    if not seed:
        random.seed(seed)
    dt = T / N

    # brownian motion 생성
    e = np.random.multivariate_normal(mean=mean, cov=cov, size=(iter_num, N))
    e[:, 0, :] = 0

    # GBM
    s = s0 * np.exp((mean - div) * dt + np.diag(cov) * np.sqrt(dt) * e.cumsum(axis=1))
    return s


def merton():
    """
    jump process
    :return:
    """

    return


def implied_distribution():
    """
    get from butterfly option portfolio
    :return:
    """

    return


def data_embedded_distribution(mean, var, skew, kurtosis, size):
    """
    To reflect population's 1st, 2nd, 3rd, 4th momentum
    :return: random sample
    """
    # from scipy import stats, optimize
    # import numpy as np
    #
    # def random_by_moment(moment, value, size):
    #     """ Draw `size` samples out of a generalised Gamma distribution
    #     where a given moment has a given value """
    #     assert moment in 'mvsk', "'{}' invalid moment. Use 'm' for mean," \
    #                              "'v' for variance, 's' for skew and 'k' for kurtosis".format(moment)
    #
    #     def gengamma_error(a):
    #         m, v, s, k = (stats.gengamma.stats(a[0], a[1], moments="mvsk"))
    #         moments = {'m': m, 'v': v, 's': s, 'k': k}
    #         return (moments[moment] - value) ** 2  # has its minimum at the desired value
    #
    #     a, c = optimize.minimize(gengamma_error, (1, 1)).x
    #     return stats.gengamma.rvs(a, c, size=size)
    #
    # n = random_by_moment('k', 3, 100000)
    # # test if result is correct
    # print("mean={}, var={}, skew={}, kurt={}".format(np.mean(n), np.var(n), stats.skew(n), stats.kurtosis(n)))
    #
    # def random_by_sk(skew, kurt, size):
    #     def gengamma_error(a):
    #         s, k = (stats.gengamma.stats(a[0], a[1], moments="sk"))
    #         return (s - skew) ** 2 + (k - kurt) ** 2  # penalty equally weighted for skew and kurtosis
    #
    #     a, c = optimize.minimize(gengamma_error, (1, 1)).x
    #     return stats.gengamma.rvs(a, c, size=size)
    #
    # n = random_by_sk(3, 3, 100000)
    # print("mean={}, var={}, skew={}, kurt={}".format(np.mean(n), np.var(n), stats.skew(n), stats.kurtosis(n)))
    # # will yield skew ~2 and kurtosis ~3 instead of 3, 3
    return
