# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 10. 29
"""
from copy import deepcopy
from tqdm import tqdm
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy.stats import iqr, f_oneway
from multiprocessing import Pool

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
import tensorflow as tf
from keras import backend as k

from columns import *
from Black_Sholes.bs_formula import *
from volatility.DNN_model_vol import DNN_vol
from volatility.LSTM_model_vol import GRU_vol

warnings.filterwarnings(action='ignore')
# read data
# KS200, CD91, dividend
underlying = pd.read_excel('data/underlying.xlsx', sheet_name='Sheet2')
dividend = underlying[[TRADE_DATE, KS200_DIV]]
dividend[TRADE_DATE] = pd.to_datetime(dividend[TRADE_DATE], format='%Y. %m')
dividend[KS200_DIV] *= 0.01
dividend.set_index(TRADE_DATE, inplace=True)
dividend = dividend.resample('D').last()
dividend.interpolate(method='linear', inplace=True)
dividend.reset_index(inplace=True)
underlying[KS200_DIV] = dividend[KS200_DIV]
underlying[CD91] *= 0.01
underlying[KS200_CHG] = underlying[KS200] - underlying[KS200].shift(1)
underlying[KS200_RET] = underlying[KS200_CHG] / underlying[KS200]
underlying[KS200 + HIST_VOL] = underlying[KS200_RET].rolling(60).std()
underlying[KS200 + HIST_VOL + '_chg'] = underlying[KS200 + HIST_VOL] - underlying[KS200 + HIST_VOL].shift(1)
underlying[CD91 + '_chg'] = underlying[CD91] - underlying[CD91].shift(1)

# read option data
options = pd.read_csv('data/option/KS200_option.csv')
options[TRADE_DATE] = pd.to_datetime(options[TRADE_DATE], format='%Y-%m-%d')
options[TOW] /= 365

# volatility of volatility
vkospi = pd.read_csv('data/option/VKOSPI(2013.08.06~2019.11.01_).csv')
vkospi.rename(columns={'TRADE_DATE': TRADE_DATE, 'VKOSPI': VKOSPI}, inplace=True)
vkospi[TRADE_DATE] = pd.to_datetime(vkospi[TRADE_DATE], format='%Y-%m-%d')
vkospi.sort_values(TRADE_DATE, inplace=True)

# setting data
# merge data
total_data = pd.merge(options, vkospi, how='left', on=[TRADE_DATE])
total_data = pd.merge(total_data, underlying, how='left', on=[TRADE_DATE])

# calculate change of option information
total_data.sort_values([MAT, K, CP], inplace=True)
total_data.reset_index(drop=True, inplace=True)
total_data['imp_vol_chg'] = total_data.groupby([MAT, K, CP]).apply(
    lambda x: (x[IMPLIED_VOL] - x[IMPLIED_VOL].shift(1))).reset_index(drop=True)

total_data.sort_values([MAT, K, CP], inplace=True)
total_data.reset_index(drop=True, inplace=True)
total_data[MKT_PRICE_CHG] = total_data.groupby([MAT, K, CP]).apply(
    lambda x: x[CLOSE_P] - x[CLOSE_P].shift(1)).reset_index(drop=True)

total_data.sort_values([MAT, K, CP], inplace=True)
total_data.reset_index(drop=True, inplace=True)
total_data['ttm_chg'] = total_data.groupby([MAT, K, CP]).apply(
    lambda x: x[TOW] - x[TOW].shift(1)).reset_index(
    drop=True)

# greek
total_data['bs_' + DELTA] = bs_delta(total_data[KS200], total_data[K], total_data[TOW], total_data[CD91],
                                     total_data[KS200_DIV], total_data[IMPLIED_VOL], 1, total_data[CP])
total_data['bs_' + GAMMA] = bs_gamma(total_data[KS200], total_data[K], total_data[TOW], total_data[CD91],
                                     total_data[KS200_DIV], total_data[IMPLIED_VOL], 1)
total_data['bs_' + VEGA] = bs_vega(total_data[KS200], total_data[K], total_data[TOW], total_data[CD91],
                                   total_data[KS200_DIV], total_data[IMPLIED_VOL], 1)
total_data['bs_' + THETA] = bs_theta(total_data[KS200], total_data[K], total_data[TOW], total_data[CD91],
                                     total_data[KS200_DIV], total_data[IMPLIED_VOL], 1, total_data[CP])
total_data['bs_' + RHO] = bs_rho(total_data[KS200], total_data[K], total_data[TOW], total_data[CD91],
                                 total_data[KS200_DIV], total_data[IMPLIED_VOL], 1, total_data[CP])


# total_data.dropna(subset=[VKOSPI, KS200_F], inplace=True)

# using func
def get_params(train_set):
    y = (train_set[MKT_PRICE_CHG] - train_set['bs_' + DELTA] * train_set[KS200_CHG]) / train_set['bs_' + VEGA]

    a_ = train_set[KS200_RET] / np.sqrt(train_set[TOW])
    b_ = train_set['bs_' + DELTA] * (train_set[KS200_RET] / np.sqrt(train_set[TOW]))
    c_ = (train_set['bs_' + DELTA] ** 2) * (train_set[KS200_RET] / np.sqrt(train_set[TOW]))

    x = pd.DataFrame(np.array([a_.values, b_.values, c_.values]).T, columns=['a', 'b', 'c'])
    model = LinearRegression().fit(x, y)
    return model.coef_


def filtering(unrefined_total_data, cp, method, JD=False):
    unrefined_total_data = unrefined_total_data[unrefined_total_data[CP] == cp]
    unrefined_total_data = unrefined_total_data[unrefined_total_data[KS200_RET] != 0]
    # delta constrain
    if JD:
        unrefined_total_data['hist_' + GAMMA] = bs_gamma(unrefined_total_data[KS200], unrefined_total_data[K],
                                                         unrefined_total_data[TOW], unrefined_total_data[CD91],
                                                         unrefined_total_data[KS200_DIV],
                                                         unrefined_total_data[KS200 + HIST_VOL], 1)
        unrefined_total_data['hist_' + VEGA] = bs_vega(unrefined_total_data[KS200], unrefined_total_data[K],
                                                       unrefined_total_data[TOW], unrefined_total_data[CD91],
                                                       unrefined_total_data[KS200_DIV],
                                                       unrefined_total_data[KS200 + HIST_VOL], 1)
        unrefined_total_data['hist_' + THETA] = bs_theta(unrefined_total_data[KS200], unrefined_total_data[K],
                                                         unrefined_total_data[TOW], unrefined_total_data[CD91],
                                                         unrefined_total_data[KS200_DIV],
                                                         unrefined_total_data[KS200 + HIST_VOL], 1,
                                                         unrefined_total_data[CP])
        unrefined_total_data['hist_' + RHO] = bs_rho(unrefined_total_data[KS200], unrefined_total_data[K],
                                                     unrefined_total_data[TOW], unrefined_total_data[CD91],
                                                     unrefined_total_data[KS200_DIV],
                                                     unrefined_total_data[KS200 + HIST_VOL], 1,
                                                     unrefined_total_data[CP])
        unrefined_total_data['real_' + DELTA] = (unrefined_total_data[MKT_PRICE_CHG] - (
                unrefined_total_data['hist_' + GAMMA] * (unrefined_total_data[KS200_CHG] ** 2) / 2 +
                unrefined_total_data['hist_' + THETA] * unrefined_total_data['ttm_chg'] + unrefined_total_data[
                    'hist_' + VEGA] * unrefined_total_data[KS200 + HIST_VOL + '_chg'] + unrefined_total_data[
                    CD91 + '_chg'] * unrefined_total_data['hist_' + RHO])) / unrefined_total_data[KS200_CHG]
        unrefined_total_data = unrefined_total_data[
            (unrefined_total_data[CP] * unrefined_total_data['real_' + DELTA] >= 0.05) & (
                    unrefined_total_data[CP] * unrefined_total_data['real_' + DELTA] <= 0.95)]
    else:
        unrefined_total_data = unrefined_total_data[
            (unrefined_total_data[CP] * unrefined_total_data['bs_' + DELTA] <= 0.95) & (
                    unrefined_total_data[CP] * unrefined_total_data['bs_' + DELTA] >= 0.05)]
    # tow, imp vol, volume constrain
    unrefined_total_data = unrefined_total_data[unrefined_total_data[TOW] >= 7 / 365]
    unrefined_total_data = unrefined_total_data[unrefined_total_data[IMPLIED_VOL] > 0.03]
    unrefined_total_data = unrefined_total_data[unrefined_total_data[TRADING_VOLUME] >= 10]

    # additional constrain for NN
    if method == 'DNN' or 'LSTM':
        lower = unrefined_total_data[['imp_vol_chg']].describe().iloc[4][0]
        upper = unrefined_total_data[['imp_vol_chg']].describe().iloc[6][0]
        iqr_ = iqr(unrefined_total_data['imp_vol_chg'].values, rng=(25, 75))
        unrefined_total_data = unrefined_total_data[
            (unrefined_total_data['imp_vol_chg'] > lower - iqr_) & (unrefined_total_data['imp_vol_chg'] < upper + iqr_)]
    return unrefined_total_data


def load_set(refined_total_data, total_trade_dates, date, TESTING_PERIOD, time_step=None, method=None, freq=None):
    if method in ['ridge_reg', 'SVR', 'DNN']:
        train_set, test_set, y_train, y_test = train_test_split(refined_total_data, refined_total_data['imp_vol_chg'],
                                                                test_size=0.1, random_state=42)
        return train_set, test_set, y_train, y_test
    elif method == 'LSTM':
        refined_total_data.reset_index(drop=True, inplace=True)
        train_set = refined_total_data.iloc[: int(len(refined_total_data) * 0.9), :][
            [MAT, K, KS200_RET, 'bs_' + DELTA, TOW, 'bs_' + VEGA, MKT_PRICE_CHG, KS200_CHG, KS200, 'imp_vol_chg']]
        test_set = refined_total_data.iloc[int(len(refined_total_data) * 0.9):, :][
            [MAT, K, KS200_RET, 'bs_' + DELTA, TOW, 'bs_' + VEGA, MKT_PRICE_CHG, KS200_CHG, KS200, 'imp_vol_chg']]

        # preprocess sequence data
        train_option_data_list = dict()
        train_options_list = []
        for option, option_data in train_set.groupby([MAT, K]):
            if len(option_data) >= time_step:
                train_option_data_list[option] = option_data.values
                train_options_list.append(option)

        test_option_data_list = dict()
        test_options_list = []
        for option, option_data in test_set.groupby([MAT, K]):
            if option_data.shape[0] == time_step:
                test_option_data_list[option] = option_data.values
                test_options_list.append(option)

        _temp_train_list = []
        for option in train_options_list:
            option_data = train_option_data_list[option]
            for i in range(option_data.shape[0] - time_step + 1):
                _temp_train_list.append(option_data[i:i + time_step])
        data_train_array = np.array(_temp_train_list)

        _temp_test_list = []
        for option in test_options_list:
            option_data = test_option_data_list[option]
            _temp_test_list.append(option_data)
        data_test_array = np.array(_temp_test_list)

        for index, row in train_set.iterrows():
            if (row[MAT], row[K]) not in train_options_list:
                train_set = train_set.drop(index, 0)

        for index, row in test_set.iterrows():
            if (row[MAT], row[K]) not in test_options_list:
                test_set = test_set.drop(index, 0)
        test_set = test_set.groupby([MAT, K]).last().reset_index()
        return train_set, test_set, data_train_array, data_test_array
    else:
        train_start_date = total_trade_dates.iloc[np.where(total_trade_dates == date)[0][0] - TESTING_PERIOD - 1][0]
        train_end_date = total_trade_dates.iloc[np.where(total_trade_dates == date)[0][0] - 1][0]
        train_set = refined_total_data.loc[
            (refined_total_data[TRADE_DATE] >= train_start_date) & (refined_total_data[TRADE_DATE] <= train_end_date)]
        if freq == 'M':
            test_end_date = total_trade_dates.iloc[np.where(total_trade_dates == date)[0][0] + 19][0]
            test_set = refined_total_data.loc[
                (refined_total_data[TRADE_DATE] >= date) & (refined_total_data[TRADE_DATE] <= test_end_date)]
        else:
            test_set = refined_total_data.loc[refined_total_data[TRADE_DATE] == date]
        return train_set, test_set, 0, 0


def get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, date, TESTING_PERIOD, method, surface=None,
                           freq=None, info=None):
    if method == 'empirical_reg':
        train_set, test_set, _, _ = load_set(refined_total_data, total_trade_dates, date, TESTING_PERIOD, method=method,
                                             freq=freq)
        params = get_params(train_set)
        test_set['E_imp_chg_{}'.format(method)] = test_set[KS200_RET] * (
                params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (test_set['bs_' + DELTA] ** 2)) / (
                                                      np.sqrt(test_set[TOW]))
        test_set['MV_delta_{}'.format(method)] = test_set['bs_' + DELTA] + test_set['bs_' + VEGA] * (
                params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (test_set['bs_' + DELTA] ** 2)) / (
                                                         test_set[KS200] * np.sqrt(test_set[TOW]))
        return test_set

    if method == 'local_vol':
        _, test_set, _, _ = load_set(refined_total_data, total_trade_dates, date, TESTING_PERIOD, method=method,
                                     freq=freq)
        _skew = surface.groupby('EXPMM').apply(lambda x: (x[LOCAL_VOL] - x[LOCAL_VOL].shift(-1)) / (
                (x[MONEYNESS] - x[MONEYNESS].shift(-1)) * 2)).reset_index()
        _skew.set_index('level_1', inplace=True)
        del _skew['EXPMM']
        surface['local_vol_skew'] = _skew.fillna(method='ffill')
        test_set['local_vol_skew'] = surface['local_vol_skew']
        test_set['MV_delta_{}'.format(method)] = test_set['bs_' + DELTA] + test_set['bs_' + VEGA] * test_set[
            'local_vol_skew']
        return test_set

    if method == 'ridge_reg':
        # 훈련 데이터 구성
        train_set, test_set, _, _ = load_set(refined_total_data, total_trade_dates, None, TESTING_PERIOD, method=method,
                                             freq=freq)
        train_set = train_set[train_set[KS200_CHG] != 0]
        train_set['real_delta'] = train_set[MKT_PRICE_CHG] / train_set[KS200_CHG]
        train_set[MONEYNESS] = train_set[K] / train_set[KS200]
        test_set[MONEYNESS] = test_set[K] / test_set[KS200]
        x_train = train_set[[MONEYNESS, TOW, 'bs_' + DELTA]]
        y_train = train_set['real_delta']
        x_test = test_set[[MONEYNESS, TOW, 'bs_' + DELTA]]

        # 훈련 데이터 정규화
        scaler = StandardScaler()
        scaler.fit(x_train)
        X_train2 = scaler.transform(x_train)
        X_train2 = pd.DataFrame(X_train2)
        X_train2.columns = x_train.iloc[:, :].columns
        x_train = X_train2

        # Ridge + Cross validation
        kr = Ridge()
        param = {'alpha': [10 ** (-7), 10 ** (-6), 10 ** (-5), 10 ** (-4), 10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** 0,
                           10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]}
        grid_ridge = GridSearchCV(kr, param_grid=param, cv=5, scoring='neg_mean_squared_error', refit=True)
        grid_ridge.fit(x_train, y_train)
        ridge_update = grid_ridge.best_estimator_

        # 테스트
        scaler = StandardScaler()
        scaler.fit(x_test)
        X_test2 = scaler.transform(x_test)
        X_test2 = pd.DataFrame(X_test2)
        X_test2.columns = x_test.iloc[:, :].columns
        x_test = X_test2
        y_pred_ridge = ridge_update.predict(x_test)
        test_set['MV_delta_{}'.format(method)] = y_pred_ridge
        test_set['SSE_{}'.format(method)] = (test_set[MKT_PRICE_CHG] - test_set['MV_delta_{}'.format(method)] *
                                             test_set[KS200_RET]) ** 2
        test_set['SSE_BS'] = (test_set[MKT_PRICE_CHG] - test_set['bs_' + DELTA] * test_set[KS200_RET]) ** 2
        return test_set

    if method == 'SVR':
        _df = deepcopy(refined_total_data)
        _df[VKOSPI] = refined_total_data[VKOSPI].shift(1)
        _df.dropna(inplace=True)
        train_set, test_set, _, _ = load_set(_df, total_trade_dates, None, TESTING_PERIOD, method=method, freq=freq)
        x_train = train_set[[KS200_RET, 'bs_' + DELTA, TOW, VKOSPI]].values
        y_train = train_set['imp_vol_chg'].values
        x_test = test_set[[KS200_RET, 'bs_' + DELTA, TOW, VKOSPI]].values

        # scaling
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        scaler = StandardScaler()
        x_test = scaler.fit_transform(x_test)

        # rbf
        model_poly = SVR(kernel='rbf', gamma='scale', C=info[0], epsilon=0.01)
        # free parameter: C, epsilon --> 후에 조정하며 확인할 것
        d_imp_vol = model_poly.fit(x_train, y_train).predict(x_test)
        test_set['E_imp_chg_{}'.format(method)] = pd.Series(d_imp_vol.reshape(-1), index=test_set.index)
        params = get_params(train_set)
        test_set['E_imp_chg_{}'.format('empirical_reg')] = test_set[KS200_RET] * (
                params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (test_set['bs_' + DELTA] ** 2)) / (
                                                               np.sqrt(test_set[TOW]))
        return test_set

    if method in ['DNN', 'LSTM']:
        # input parameter
        hidden_layers = info[2]
        activation = info[0]
        epochs = 4000
        batch_size = info[3]  # 모델을 학습할 때, 한 iteration(forward - backward 반복 횟수) 당 사용되는 set의 크기
        bias_initializer = 'he_uniform'
        kernel_initializer = 'glorot_uniform'
        if method == 'DNN':
            _df = deepcopy(refined_total_data)
            _df[VKOSPI] = refined_total_data[VKOSPI].shift(1)
            # _df.dropna(inplace=True)
            train_set, test_set, y_train, y_test = load_set(_df, total_trade_dates, None, None, method=method,
                                                            freq=freq)
            # prepare data
            scaler = StandardScaler()
            x_train = train_set[[KS200_RET, 'bs_' + DELTA, TOW, VKOSPI]].values
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            scaler = StandardScaler()
            x_test = test_set[[KS200_RET, 'bs_' + DELTA, TOW, VKOSPI]].values
            scaler.fit(x_test)
            x_test = scaler.transform(x_test)
            input_dim = x_train.shape[1]
            model = DNN_vol(input_dim, batch_size, epochs, activation, bias_initializer, kernel_initializer, x_train,
                            y_train, hidden_layers, lr=info[1], bias_regularizer=None, dropout=False,
                            dropout_rate=0.1, batch_normalization=True, early_stop=True)
            test_set['E_imp_chg_{}'.format(method)] = model.predict(x_test, verbose=0).reshape(-1)
            test_set['MV_delta_{}'.format(method)] = test_set['bs_' + DELTA] + test_set['bs_' + VEGA] * test_set[
                'E_imp_chg_{}'.format(method)] / test_set[KS200_RET]

            # empirical reg model
            params = get_params(train_set)
            test_set['E_imp_chg_empirical_reg'] = test_set[KS200_RET] * (
                    params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (
                    test_set['bs_' + DELTA] ** 2)) / (np.sqrt(test_set[TOW]))
            test_set['MV_delta_empirical_reg'] = test_set['bs_' + DELTA] + test_set['bs_' + VEGA] * test_set[
                'E_imp_chg_empirical_reg'] / test_set[KS200_RET]

            # calculate hedge errors
            test_set['SSE_BS'] = (test_set[MKT_PRICE_CHG] - test_set['bs_' + DELTA] * test_set[KS200_RET]) ** 2
            test_set['SSE_{}'.format(method)] = (test_set[MKT_PRICE_CHG] - test_set[
                'MV_delta_{}'.format(method)] * test_set[KS200_RET]) ** 2
            test_set['SSE_empirical_reg'] = (test_set[MKT_PRICE_CHG] - test_set['MV_delta_empirical_reg'] * test_set[
                KS200_RET]) ** 2
            test_set['y_test'] = y_test
            # Clean up the memory
            k.clear_session()
            tf.reset_default_graph()
            return test_set

        if method == 'LSTM':
            time_step = 5
            _df = deepcopy(refined_total_data)
            _df[VKOSPI] = refined_total_data[VKOSPI].shift(1)
            train_set, test_set, data_train_array, data_test_array = load_set(_df, total_trade_dates, None,
                                                                              TESTING_PERIOD, time_step, method, freq)
            # setting set
            x_train = data_train_array[:, :, 2:-5]  # KS200_F_RET, DELTA, TOW, VKOSPI
            y_train = data_train_array[:, :, [-1]]  # imp_vol_chg
            x_test = data_test_array[:, :, 2:-5]

            # # scaling
            # scaler = StandardScaler()
            # scaler.fit(x_train)
            # x_train = scaler.transform(x_train)
            # scaler = StandardScaler()
            # scaler.fit(x_test)
            # x_test = scaler.transform(x_test)

            # model training
            input_dim = x_train.shape[2]
            model = GRU_vol(input_dim, time_step, batch_size, epochs, activation, bias_initializer,
                            kernel_initializer, None, hidden_layers, False, 0, True, True, x_train, y_train, lr=info[1])
            test_set['E_imp_chg_{}'.format(method)] = model.predict(x_test, verbose=0)[:, -1, :].reshape(-1)
            params = get_params(train_set)
            test_set['E_imp_chg_{}'.format('empirical_reg')] = test_set[KS200_RET] * (
                    params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (test_set['bs_' + DELTA] ** 2)) / (
                                                                   np.sqrt(test_set[TOW]))
            # Clean up the memory
            k.clear_session()
            tf.reset_default_graph()
            return test_set


def make_int(n):
    return int(n * 10)


make_int = np.vectorize(make_int)


def delta_gain(results, method):
    data = deepcopy(results)
    data['bs_' + DELTA] = make_int(data['bs_' + DELTA])
    df2 = data.groupby('bs_' + DELTA).apply(lambda x: 1 - (x['SSE_{}'.format(method)].sum() / x['SSE_BS'].sum()))
    return df2


def ML_delta_gain(results, method):
    data = deepcopy(results)
    data['bs_' + DELTA] = make_int(data['bs_' + DELTA])
    df2 = data.groupby('bs_' + DELTA).apply(
        lambda x: 1 - (x['SSE_{}'.format(method)].sum() / x['SSE_empirical_reg'].sum()))
    return df2


def tow_classifier(x):
    if x < 0.5:
        return 'tow < 0.5'
    if 0.5 <= x < 1:
        return '0.5 < tow < 1'
    if 1 <= x < 2:
        return '1 < tow < 2'
    if x >= 2:
        return 'tow >= 2'


def return_classifier(x):
    if x < -0.0125:
        return 'ret < -1.25%'
    if -0.0125 <= x < 0.0125:
        return '-1.25% <= ret < 1.25%'
    if x >= 0.0125:
        return 'ret >= 1.25%'


def get_backtesting(total_data, start_date, end_date, cp, method, TESTING_PERIOD=240, freq=None, info=None):
    # constrain to delta, tow, trading volume
    unrefined_total_data = deepcopy(total_data)
    refined_total_data = filtering(unrefined_total_data, cp, method, JD=True)
    total_trade_dates = pd.DataFrame(np.unique(refined_total_data[TRADE_DATE].values), columns=[DATE])
    testing_dates = total_trade_dates[
        (total_trade_dates >= start_date) & (total_trade_dates <= end_date)].dropna().values.reshape(1, -1)[0]
    refined_total_data.sort_values([MAT, K, CP], inplace=True)

    if freq == 'M':
        _df = pd.DataFrame(testing_dates, columns=[TRADE_DATE])
        _df['YEAR'] = _df[TRADE_DATE].apply(lambda x: x.year)
        _df['MONTH'] = _df[TRADE_DATE].apply(lambda x: x.month)
        testing_dates = _df.groupby(['YEAR', 'MONTH']).first().reset_index()[TRADE_DATE].values

    if method in ['ridge_reg', 'SVR', 'DNN', 'LSTM']:
        results = get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, None, TESTING_PERIOD, method, info=info)
        results['empirical_reg_SSE'] = (results['imp_vol_chg'] - results['E_imp_chg_{}'.format('empirical_reg')]) ** 2
        results['{}_SSE'.format(method)] = (results['imp_vol_chg'] - results['E_imp_chg_{}'.format(method)]) ** 2
        results['ttm'] = results[TOW].apply(lambda x: tow_classifier(x))
        results['ut'] = results[KS200_RET].apply(lambda x: return_classifier(x))
        return results
    else:
        results = pd.DataFrame([])
        for date in tqdm(testing_dates):
            if method == 'local_vol':
                surface = unrefined_total_data[unrefined_total_data[TRADE_DATE] == date]
                test_set = get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, date, TESTING_PERIOD, method,
                                                  surface, freq)
            else:
                test_set = get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, date, TESTING_PERIOD, method,
                                                  freq=freq)
            test_set['SSE_{}'.format(method)] = (test_set[MKT_PRICE_CHG] - test_set['MV_delta_{}'.format(method)] *
                                                 test_set[KS200_CHG] * 10) ** 2
            test_set['SSE_BS'] = (test_set[MKT_PRICE_CHG] - test_set['bs_' + DELTA] * test_set[KS200_CHG] * 10) ** 2
            test_set['ttm'] = test_set[TOW].apply(lambda x: tow_classifier(x))
            test_set['ut'] = test_set[KS200_RET].apply(lambda x: return_classifier(x))
            # test_set['real_delta'] = test_set[MKT_PRICE_CHG] / test_set[KS200_F_CHG]
            results = pd.concat([results, test_set], ignore_index=True, axis=0)
        return results


call_emp_reg_gain = get_backtesting(
    total_data,
    start_date=datetime(2007, 1, 15),
    end_date=datetime(2019, 12, 30),
    cp=1,
    method='empirical_reg',
    TESTING_PERIOD=240,
    freq='M'
)

put_emp_reg_gain = get_backtesting(
    total_data,
    start_date=datetime(2009, 1, 15),
    end_date=datetime(2019, 12, 30),
    cp=-1,
    method='empirical_reg',
    TESTING_PERIOD=240*3,
    freq='M'
)

call_dnn_gain = get_backtesting(
    total_data,
    start_date=datetime(2006, 1, 2),
    end_date=datetime(2019, 12, 30),
    cp=1,
    method='DNN',
    info=['tanh', 0.0003, list([80, 80, 80]), 500]
)

put_dnn_gain = get_backtesting(
    total_data,
    start_date=datetime(2006, 1, 2),
    end_date=datetime(2019, 12, 30),
    cp=-1,
    method='DNN',
    info=['tanh', 0.0003, list([80, 80, 80]), 500]
)


# plt.scatter(long_call_dnn_gain[MONEYNESS], long_call_dnn_gain['real_delta'], label='real')
# plt.scatter(long_call_dnn_gain[MONEYNESS], long_call_dnn_gain['MV_delta_DNN'], label='MV')
# plt.scatter(long_call_dnn_gain[MONEYNESS], long_call_dnn_gain['delta'], label='BS')
# plt.legend()
# plt.show()

c_e = delta_gain(call_emp_reg_gain, 'empirical_reg')
p_e = delta_gain(put_emp_reg_gain, 'empirical_reg')
# c_r = delta_gain(call_ridge_reg_gain, 'ridge_reg')
# p_r = delta_gain(put_ridge_reg_gain, 'ridge_reg')
# c_re = ML_delta_gain(call_ridge_reg_gain, 'ridge_reg')
# p_re = ML_delta_gain(put_ridge_reg_gain, 'ridge_reg')
c_d = delta_gain(call_dnn_gain, 'DNN')
c_de = ML_delta_gain(call_dnn_gain, 'DNN')
p_d = delta_gain(put_dnn_gain, 'DNN')
p_de = ML_delta_gain(put_dnn_gain, 'DNN')
# c_l = delta_gain(call_lstm_gain, 'LSTM')
# p_l = delta_gain(put_lstm_gain, 'LSTM')
# c_le = ML_delta_gain(call_lstm_gain, 'LSTM')
# p_le = ML_delta_gain(put_lstm_gain, 'LSTM')

separated_BS = pd.pivot_table(call_dnn_gain, values='SSE_DNN', columns=['ttm'], index=['ut'], aggfunc=np.sum)
separated_MV = pd.pivot_table(call_dnn_gain, values='SSE_empirical_reg', columns=['ttm'], index=['ut'], aggfunc=np.sum)
a = 1 - separated_BS / separated_MV

anova_analysis = f_oneway(call_dnn_gain['hedge_error_empirical_reg'].values,
                          # call_ridge_reg_gain['hedge_error_ridge_reg'].values,
                          call_dnn_gain['hedge_error_DNN'].values)

if __name__ == '__main__':
    core_num = 15
    # activation_func = ['sigmoid', 'tanh', 'relu', 'swish']
    #     # learning_rate = [0.01, 0.001, 0.0001, 0.03, 0.003, 0.0003]
    #     # hidden_layer = [[30], [50], [80], [30, 30, 30], [50, 50, 50], [80, 80, 80]]
    #     # batch_size = [50, 100, 300, 500, 1000]
    #     #
    #     # lis = [activation_func, learning_rate, hidden_layer, batch_size]
    #     #
    #     # grid = np.array([])
    #     # for a in range(len(lis[0])):
    #     #     for b in range(len(lis[1])):
    #     #         for c in range(len(lis[2])):
    #     #             for d in range(len(lis[3])):
    #     #                 grid = np.concatenate([grid, [lis[0][a], lis[1][b], lis[2][c], lis[3][d]]], axis=0)
    #     # grid = grid.reshape((720, 4))
    with Pool(core_num) as p:
        results = [p.apply_async(get_backtesting, t) for t in
                   #               [total_data, datetime(2011, 1, 5), datetime(2019, 5, 28), 1, 'DNN', 200 * 3, 'M', t])
                   # for t in grid]
                   [
                       [total_data, datetime(2017, 1, 5), datetime(2019, 5, 31), 1, 1, 'SVR', 240, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                       [total_data, datetime(201, 1, 5), datetime(2019, 5, 31), -1, 1, 'SVR', 240, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                       [total_data, datetime(2011, 1, 5), datetime(2019, 5, 28), 1, 'DNN', 200 * 3, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                       [total_data, datetime(2011, 1, 5), datetime(2019, 5, 31), -1, 'DNN', 200 * 3, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                       [total_data, datetime(2017, 1, 5), datetime(2019, 5, 31), 1, 'LSTM', 240, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                       [total_data, datetime(2017, 1, 5), datetime(2019, 5, 31), -1, 'LSTM', 240, 'M',
                        ['tanh', 0.0003, list([80, 80, 80]), 500]],
                   ]]
        for r in results:
            r.wait()
        results = [result.get() for result in results]
        for idx, result in enumerate(results):
            result.to_csv('data/{}.csv'.format(idx), index=False)
        p.close()
        p.join()
