# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 09. 27
"""
from copy import deepcopy as dc
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas.core.common as com
from pandas.core.index import MultiIndex
from pandas.core.indexing import convert_to_index_sliceable
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from Black_Sholes.bs_formula import *
from columns import *
from stock_process import GBM


class Portfolio(DataFrame):
    """
    portfolio for options
    """
    @property
    def _constructor(self):
        return Portfolio

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):
        DataFrame.__init__(self=self, data=data, index=index, columns=columns, dtype=dtype, copy=copy)

    def to_dataframe(self, deepcopy: bool = True) -> DataFrame:
        """
        Convert portfolio to dataframe type.
        :param deepcopy : (bool) If deepcopy is True, convert to dataframe based on deepcopy. Or, convert to dataframe
                                  based on shallow copy.
        :return dataframe : (DataFrame) Converted dataframe type portfolio
        """
        if deepcopy:
            dataframe = DataFrame(dc(self))
        else:
            dataframe = DataFrame(self)

        return dataframe

    def get_greeks(self):
        # calculate greeks for each options
        self[DELTA] = bs_delta(self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[LONG_SHORT], self[CP])
        self[GAMMA] = bs_gamma(self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[LONG_SHORT])
        self[THETA] = bs_theta(self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[LONG_SHORT], self[CP])
        self[VEGA] = bs_vega(self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[LONG_SHORT])
        self[RHO] = bs_rho(self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[LONG_SHORT], self[CP])
        self[IMPLIED_VOL] = IV(self[S], self[K], self[TOW], self[R], self[Q], self[CLOSE_P], self[CP])

        portfolio_greeks = {
            DELTA: self[DELTA].sum(),
            GAMMA: self[GAMMA].sum(),
            THETA: self[THETA].sum(),
            VEGA: self[VEGA].sum(),
            RHO: self[RHO].sum()
        }
        return portfolio_greeks

    def get_value_and_payoff(self, x=S, y=None):
        """
        COL = [S, K, TOW, R, Q, VOL, MKT_PRICE, LONG_SHORT, CP]
        :param x:
        :param y:
        :return:
        """
        # 2-dim
        split_number = 100
        if y is None:
            if x == S:
                stock_price = np.linspace(min(self[K]) * 0.75, max(self[K]) * 1.25, split_number)
                payoff_vector = np.zeros(split_number)
                value_vector = np.zeros(split_number)
                for i in range(len(self)):
                    if self[CP][i] == 1:  # call
                        payoff_vector += self[LONG_SHORT][i] * np.max(
                            np.array([stock_price - self[K][i], np.zeros(split_number)]), axis=0)
                        s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                        value_vector += self[LONG_SHORT][i] * bs_price(stock_price, k, tow, r, q, vol, cp)
                    else:  # put
                        payoff_vector += self[LONG_SHORT][i] * np.max(
                            np.array([self[K][i] - stock_price, np.zeros(split_number)]), axis=0)
                        s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                        value_vector += self[LONG_SHORT][i] * bs_price(stock_price, k, tow, r, q, vol, cp)
                plt.plot(stock_price, payoff_vector)
                plt.plot(stock_price, value_vector)

            if x == TOW:
                TTM = np.linspace(min(self[TOW]), 1E-14, split_number)
                value_vector = np.zeros(split_number)
                for i in range(len(self)):
                    s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                    value_vector += self[LONG_SHORT][i] * bs_price(s, k, TTM, r, q, vol, cp)
                plt.plot(TTM, value_vector)

            if x == R:
                R_value = np.linspace(0.0001, 0.15, split_number)
                value_vector = np.zeros(split_number)
                for i in range(len(self)):
                    s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                    value_vector += self[LONG_SHORT][i] * bs_price(s, k, tow, R_value, q, vol, cp)
                plt.plot(R_value, value_vector)

            if x == Q:
                Q_value = np.linspace(0.0001, 0.15, split_number)
                value_vector = np.zeros(split_number)
                for i in range(len(self)):
                    s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                    value_vector += self[LONG_SHORT][i] * bs_price(s, k, tow, r, Q_value, vol, cp)
                plt.plot(Q_value, value_vector)

            if x == VOL:
                vol_value = np.linspace(min(self[VOL]) * 0.5, max(self[VOL]) * 2, split_number)
                value_vector = np.zeros(split_number)
                for i in range(len(self)):
                    s, k, tow, r, q, vol, mkt, ls, cp, *el = self.iloc[i]
                    value_vector += self[LONG_SHORT][i] * bs_price(s, k, tow, r, q, vol_value, cp)
                plt.plot(vol_value, value_vector)

            plt.show()

        # 3-dim
        else:
            if x == TOW or y == TOW:
                if x == TOW:
                    # x가 TOW인 경우
                    col_names = [S, K, TOW, R, Q, VOL, CP]
                    cols = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                    x_range = np.linspace(0, self[x].unique(), split_number)
                    y_range = np.linspace(min(self[y]) * 0.75, max(self[y]) * 1.25, split_number)
                    x_range, y_range = np.meshgrid(x_range, y_range)
                    x_index = col_names.index(x)
                    y_index = col_names.index(y)
                    cols[x_index] = x_range
                    cols[y_index] = y_range
                    value_vector = np.zeros((split_number, split_number))
                    for i in range(split_number):
                        for j in range(split_number):
                            x_value = cols[x_index][i, j]
                            y_value = cols[y_index][i, j]
                            input_list = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                            input_list[x_index] = x_value
                            input_list[y_index] = y_value
                            port_value = bs_price(input_list[0], input_list[1], input_list[2], input_list[3],
                                                  input_list[4],
                                                  input_list[5], input_list[6])
                            port_value = port_value.sum()
                            value_vector[i, j] = port_value
                else:
                    # y가 TOW 인 경우
                    col_names = [S, K, TOW, R, Q, VOL, CP]
                    cols = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                    x_range = np.linspace(min(self[x]) * 0.75, max(self[x]) * 1.25, split_number)
                    y_range = np.linspace(0, self[y].unique(), split_number)
                    x_range, y_range = np.meshgrid(x_range, y_range)
                    x_index = col_names.index(x)
                    y_index = col_names.index(y)
                    cols[x_index] = x_range
                    cols[y_index] = y_range
                    value_vector = np.zeros((split_number, split_number))
                    for i in range(split_number):
                        for j in range(split_number):
                            x_value = cols[x_index][i, j]
                            y_value = cols[y_index][i, j]
                            input_list = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                            input_list[x_index] = x_value
                            input_list[y_index] = y_value
                            port_value = bs_price(input_list[0], input_list[1], input_list[2], input_list[3],
                                                  input_list[4],
                                                  input_list[5], input_list[6])
                            port_value = port_value.sum()
                            value_vector[i, j] = port_value
            else:
                col_names = [S, K, TOW, R, Q, VOL, CP]
                cols = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                x_range = np.linspace(min(self[x]) * 0.75, max(self[x]) * 1.25, split_number)
                y_range = np.linspace(min(self[y]) * 0.75, max(self[y]) * 1.25, split_number)
                x_range, y_range = np.meshgrid(x_range, y_range)
                x_index = col_names.index(x)
                y_index = col_names.index(y)
                cols[x_index] = x_range
                cols[y_index] = y_range
                value_vector = np.zeros((split_number, split_number))
                for i in range(split_number):
                    for j in range(split_number):
                        x_value = cols[x_index][i, j]
                        y_value = cols[y_index][i, j]
                        input_list = [self[S], self[K], self[TOW], self[R], self[Q], self[VOL], self[CP]]
                        input_list[x_index] = x_value
                        input_list[y_index] = y_value
                        port_value = bs_price(input_list[0], input_list[1], input_list[2], input_list[3], input_list[4],
                                              input_list[5], input_list[6])
                        port_value = port_value.sum()
                        value_vector[i, j] = port_value
            # Plot the surface.
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(x_range, y_range, value_vector, cmap=plt.cm.viridis, linewidth=0.2,
                                   antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            ax.set_zlabel(VALUE)
            plt.show()
            return value_vector

    def get_scenario(self, min_percent: float = -0.3, max_percent: float = 0.3):
        underlying_yields = np.linspace(min_percent, max_percent, num=int((max_percent - min_percent) * 100) + 1,
                                        endpoint=True) + 1
        for _ in range(len(self[S])):
            underlying_yields.append((np.linspace(min_percent, max_percent,
                                                  num=int((max_percent - min_percent) * 100) + 1,
                                                  endpoint=True) + 1).tolist())
        underlying_yields = np.array(underlying_yields)
        underlying_scenario = underlying_yields * self[S].reshape(1, len(self[S]))

        return
