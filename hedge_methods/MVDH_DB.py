# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 02. 14
"""
from copy import deepcopy
from tqdm import tqdm
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.stats import iqr, f_oneway
from scipy.stats import ranksums
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from multiprocessing import Pool
import tensorflow as tf
from keras import backend as k

from columns import *
from Black_Sholes.bs_formula import *
from volatility.DNN_model_vol import DNN_vol

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
IV = options.groupby([TRADE_DATE]).mean()[['implied_volatility']]

# volatility of volatility
vkospi = pd.read_csv('data/option/VKOSPI(2013.08.06~2019.11.01_).csv')
vkospi.rename(columns={'TRADE_DATE': TRADE_DATE, 'VKOSPI': VKOSPI}, inplace=True)
vkospi[TRADE_DATE] = pd.to_datetime(vkospi[TRADE_DATE], format='%Y-%m-%d')
vkospi.sort_values(TRADE_DATE, inplace=True)
vkospi[VKOSPI] = vkospi.shift(1)[VKOSPI].values

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


def coint_analyzer(x, y, p):
    finding = []
    for i in range(x.shape[1]):
        score, pvalue, _ = coint(x.iloc[:, i], y)
        if pvalue < p:
            finding.append(x.columns[i])
    print(finding)
    return x[finding]


def PCA_data(x, y, c_num=2, p=0.1):
    selected_data = coint_analyzer(x, y, p)
    pca = PCA(n_components=c_num)
    pca.fit(selected_data)
    pca_data = pca.transform(selected_data)
    pca_data = pd.DataFrame(pca_data)
    return pca_data


macro = pd.read_excel('data/macro/macro.xlsx', sheet_name="Sheet2")
macro['credit_spread'] = macro['시장금리:회사채(무보증3년BBB-)(%)'] - macro['시장금리:국고3년(국채관리기금채3년)(%)']
macro['us_spread'] = macro['국채금리_미국국채(10년)(%)'] - macro['국채금리_미국국채(1년)(%)']
macro.set_index(TRADE_DATE, inplace=True)
macro.dropna(inplace=True)
sub_data = pd.merge(macro, IV, how='left', on=[TRADE_DATE])
sub_data.dropna(inplace=True)
__df = deepcopy(sub_data)
col = sub_data.columns
scaler = StandardScaler()
scaler.fit(__df)
__df = scaler.transform(__df)
__df = pd.DataFrame(__df)
__df.columns = col

macro_final = PCA_data(__df.iloc[:, :-1], __df.iloc[:, -1])
macro_final[TRADE_DATE] = sub_data.index
macro_final.set_index(TRADE_DATE, inplace=True)
macro_final.columns = ['pc1', 'pc2']
total_data = pd.merge(total_data, macro_final, how='left', on=[TRADE_DATE])


def filtering(unrefined_total_data, cp, JD=False):
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
    unrefined_total_data = unrefined_total_data[unrefined_total_data[TOW] > 5 / 365]
    unrefined_total_data = unrefined_total_data[unrefined_total_data[IMPLIED_VOL] > 0.03]
    unrefined_total_data = unrefined_total_data[unrefined_total_data[TRADING_VOLUME] >= 10]

    # additional constrain for NN
    unrefined_total_data['BS_delta_hedge_error'] = unrefined_total_data[MKT_PRICE_CHG] - unrefined_total_data[
        'bs_' + DELTA] * unrefined_total_data[KS200_CHG]
    lower = unrefined_total_data[['BS_delta_hedge_error']].describe().iloc[4][0]
    upper = unrefined_total_data[['BS_delta_hedge_error']].describe().iloc[6][0]
    iqr_ = iqr(unrefined_total_data['BS_delta_hedge_error'].values, rng=(25, 75))
    unrefined_total_data = unrefined_total_data[
        (unrefined_total_data['BS_delta_hedge_error'] > lower - iqr_) & (
                unrefined_total_data['BS_delta_hedge_error'] < upper + iqr_)]
    return unrefined_total_data


def load_set(refined_total_data, total_trade_dates, date, rolling=False, TESTING_PERIOD=240, freq=None):
    if rolling:
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
        return train_set, test_set
    else:
        train_set, test_set, _, _ = train_test_split(refined_total_data, refined_total_data['imp_vol_chg'],
                                                     test_size=0.1, random_state=42)
        return train_set, test_set


def get_params(train_set):
    y = train_set[MKT_PRICE_CHG] - train_set['bs_' + DELTA] * train_set[KS200_CHG]

    a_ = train_set['bs_' + VEGA] * train_set[KS200_RET] / np.sqrt(train_set[TOW])
    b_ = train_set['bs_' + VEGA] * train_set['bs_' + DELTA] * (train_set[KS200_RET] / np.sqrt(train_set[TOW]))
    c_ = train_set['bs_' + VEGA] * (train_set['bs_' + DELTA] ** 2) * (train_set[KS200_RET] / np.sqrt(train_set[TOW]))

    x = pd.DataFrame(np.array([a_.values, b_.values, c_.values]).T, columns=['a', 'b', 'c'])
    model = LinearRegression().fit(x, y)
    return model.coef_


def get_DNN_pred(cp, case_num, date, x_train, y_train, x_test, info):
    hidden_layers = info[2]
    activation = info[0]
    epochs = 4000
    batch_size = info[3]  # 모델을 학습할 때, 한 iteration(forward - backward 반복 횟수) 당 사용되는 set의 크기
    bias_initializer = 'he_uniform'
    kernel_initializer = 'glorot_uniform'
    input_dim = x_train.shape[1]
    model = DNN_vol(input_dim, batch_size, epochs, activation, bias_initializer, kernel_initializer, x_train,
                    y_train, hidden_layers, lr=info[1], bias_regularizer=None, dropout=info[4],
                    dropout_rate=info[5], batch_normalization=True, early_stop=True)
    # model.save('data/DB/DNN_models/{}_{}_{}.h5'.format(cp, np.datetime_as_string(date, unit='D'), case_num))
    y_pred = model.predict(x_test, verbose=0).reshape(-1)
    # Clean up the memory
    k.clear_session()
    tf.reset_default_graph()
    return y_pred


def get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, date=None, rolling=False, TESTING_PERIOD=240,
                           freq=None, info=None, include_VKOSPI=False, include_macro=False):
    if include_macro:
        refined_total_data.dropna(subset=['pc1', 'pc2'], inplace=True)
    if include_VKOSPI:
        refined_total_data[VKOSPI] = refined_total_data[VKOSPI].shift(1)
        refined_total_data.dropna(subset=[VKOSPI], inplace=True)
    # empirical reg
    train_set, test_set = load_set(refined_total_data, total_trade_dates, date, rolling, TESTING_PERIOD, freq=freq)
    params = get_params(train_set)
    test_set['E_imp_chg_empirical_reg'] = test_set[KS200_RET] * (
            params[0] + params[1] * test_set['bs_' + DELTA] + params[2] * (test_set['bs_' + DELTA] ** 2)) / (
                                              np.sqrt(test_set[TOW]))
    test_set['MV_delta_empirical_reg'] = test_set['bs_' + DELTA] + test_set['bs_' + VEGA] * test_set[
        'E_imp_chg_empirical_reg'] / test_set[KS200_CHG]

    # DNN
    # prepare data
    input_data_lst = ['bs_'+GAMMA, 'bs_' + DELTA, 'bs_' + THETA]
    if include_VKOSPI:
        input_data_lst = input_data_lst + [VKOSPI]
    if include_macro:
        input_data_lst = input_data_lst + ['pc1', 'pc2']
    scaler = StandardScaler()
    x_train = train_set[input_data_lst].values
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    y_train = (train_set[MKT_PRICE_CHG] - train_set['bs_' + DELTA] * train_set[KS200_CHG]).values
    scaler = StandardScaler()
    x_test = test_set[input_data_lst].values
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    # multiprocessing for ensemble
    core_num = 6
    model_num = 12
    with Pool(core_num) as p:
        results = [p.apply_async(get_DNN_pred, t) for t in zip(
            [train_set[CP].unique()[0] for _ in range(model_num)],
            range(model_num),
            [date for _ in range(model_num)],
            [x_train for _ in range(model_num)],
            [y_train for _ in range(model_num)],
            [x_test for _ in range(model_num)],
            [info for _ in range(model_num)]
        )]
        for r in results:
            r.wait()
        results = [result.get() for result in results]
        y_preds = pd.DataFrame([])
        for result in results:
            y_preds = pd.concat([y_preds, pd.DataFrame(result)], axis=1)

    test_set['E_imp_chg_DNN'] = y_preds.mean(axis=1).values
    test_set['MV_delta_DNN'] = test_set['bs_' + DELTA] + test_set['E_imp_chg_DNN'] / test_set[KS200_CHG]

    # benchmark
    test_set['SSE_BS'] = (test_set[MKT_PRICE_CHG] - test_set['bs_' + DELTA] * test_set[KS200_CHG]) ** 2
    test_set['SSE_empirical_reg'] = (test_set[MKT_PRICE_CHG] - test_set['MV_delta_empirical_reg'] * test_set[
        KS200_CHG]) ** 2
    test_set['SSE_DNN'] = (test_set[MKT_PRICE_CHG] - test_set['MV_delta_DNN'] * test_set[KS200_CHG]) ** 2
    return test_set


def tow_classifier(x):
    if x < 60 / 365:
        return 'tow < 60'
    if 60 / 365 <= x < 120 / 365:
        return '60 < tow < 120'
    if 120 / 365 <= x:
        return '120 < tow'


def delta_classifier(x, cp):
    if cp * x < 0.45:
        return 'OTM'
    if 0.45 <= cp * x <= 0.55:
        return 'ATM'
    if 0.55 < cp * x:
        return 'ITM'


def delta_classifier2(x, cp):
    if cp * x < 0.25:
        return 'OTM'
    if 0.25 <= cp * x < 0.45:
        return 'SOTM'
    if 0.45 <= cp * x <= 0.55:
        return 'ATM'
    if 0.55 < cp * x <= 0.75:
        return 'SITM'
    if 0.75 < cp * x:
        return 'ITM'


def return_classifier(x):
    if x < -0.0125:
        return 'ret < -1.25%'
    if -0.0125 <= x < 0.0125:
        return '-1.25% <= ret < 1.25%'
    if x >= 0.0125:
        return 'ret >= 1.25%'


def make_int(n):
    return int(n * 10)


make_int = np.vectorize(make_int)


def delta_gain(results, nominator, denominator):
    data = deepcopy(results)
    data['bs_' + DELTA] = make_int(data['bs_' + DELTA])
    df2 = data.groupby('bs_' + DELTA).apply(
        lambda x: 1 - (x['SSE_{}'.format(nominator)].sum() / x['SSE_{}'.format(denominator)].sum()))
    return df2


def delta_class_gain(results, nominator, denominator):
    data = deepcopy(results)
    df2 = data.groupby('delta_class').apply(
        lambda x: 1 - (x['SSE_{}'.format(nominator)].sum() / x['SSE_{}'.format(denominator)].sum()))
    return df2


def delta_imp_vol_gain(results, nominator, denominator):
    data = deepcopy(results)
    data['bs_' + DELTA] = make_int(data['bs_' + DELTA])
    return data.groupby('bs_' + DELTA).apply(
        lambda x: 1 - ((x['imp_vol_chg'] - x['E_imp_chg_{}'.format(nominator)]) ** 2).sum() / (
                (x['imp_vol_chg'] - x['E_imp_chg_{}'.format(denominator)]) ** 2).sum())


def get_backtesting(total_data, start_date, end_date, cp, rolling, TESTING_PERIOD=240, freq=None, info=None,
                    include_VKOSPI=False, include_macro=False):
    # constrain to delta, tow, trading volume
    unrefined_total_data = deepcopy(total_data)
    refined_total_data = filtering(unrefined_total_data, cp, JD=True)
    total_trade_dates = pd.DataFrame(np.unique(refined_total_data[TRADE_DATE].values), columns=[DATE])
    testing_dates = total_trade_dates[
        (total_trade_dates >= start_date) & (total_trade_dates <= end_date)].dropna().values.reshape(1, -1)[0]
    refined_total_data.sort_values([MAT, K, CP], inplace=True)

    if freq == 'M':
        _df = pd.DataFrame(testing_dates, columns=[TRADE_DATE])
        _df['YEAR'] = _df[TRADE_DATE].apply(lambda x: x.year)
        _df['MONTH'] = _df[TRADE_DATE].apply(lambda x: x.month)
        testing_dates = _df.groupby(['YEAR', 'MONTH']).first().reset_index()[TRADE_DATE].values

    if rolling:
        results = pd.DataFrame([])
        for date in tqdm(testing_dates):
            test_set = get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, date, rolling, TESTING_PERIOD,
                                              freq=freq, info=info, include_VKOSPI=include_VKOSPI,
                                              include_macro=include_macro)
            test_set['ttm'] = test_set[TOW].apply(lambda x: tow_classifier(x))
            test_set['ut'] = test_set[KS200_RET].apply(lambda x: return_classifier(x))
            test_set['delta_class'] = test_set.apply(lambda x: delta_classifier2(x['bs_' + DELTA], x[CP]), axis=1)
            results = pd.concat([results, test_set], ignore_index=True, axis=0)
        return results
    else:
        results = get_E_imp_chg_MV_delta(refined_total_data, total_trade_dates, None, rolling, TESTING_PERIOD,
                                         freq=freq, info=info, include_VKOSPI=include_VKOSPI,
                                         include_macro=include_macro)
        results['ttm'] = results[TOW].apply(lambda x: tow_classifier(x))
        results['ut'] = results[KS200_RET].apply(lambda x: return_classifier(x))
        results['delta_class'] = results.apply(lambda x: delta_classifier2(x['bs_' + DELTA], x[CP]), axis=1)
        return results


# %%
if __name__ == '__main__':
    call_total = get_backtesting(
        total_data,
        start_date=datetime(2015, 1, 15),
        end_date=datetime(2019, 12, 30),
        cp=1,
        rolling=False,
        TESTING_PERIOD=240 * 9,
        freq='M',
        info=['sigmoid', 0.0003, list([80, 80, 80, 80, 80, 80]), 500, True, 0.3],
        include_VKOSPI=False,
        include_macro=False
    )
    call_total.to_csv('data/DB/call_ensemble_last_gamma.csv', index=False, encoding='utf-8')
    print(delta_class_gain(call_total, 'empirical_reg', 'BS'))
    print(delta_class_gain(call_total, 'DNN', 'empirical_reg'))
    separated_BS = pd.pivot_table(call_total, values='SSE_DNN', columns=['delta_class'], index=['ut'], aggfunc=np.sum)
    separated_MV = pd.pivot_table(call_total, values='SSE_empirical_reg', columns=['delta_class'], index=['ut'],
                                  aggfunc=np.sum)
    print(1 - separated_BS / separated_MV)
    for idx, data in call_total.groupby('delta_class'):
        print(delta_class_gain(data, 'empirical_reg', 'BS'))
        print(idx, ranksums(data['SSE_empirical_reg'], data['SSE_BS']))

    for idx, data in call_total.groupby('delta_class'):
        print(delta_class_gain(data, 'DNN', 'empirical_reg'))
        print(idx, ranksums(data['SSE_empirical_reg'], data['SSE_DNN']))

    for idx, data in call_total.groupby(['delta_class', 'ut']):
        print(idx, 1 - data['SSE_DNN'].sum() / data['SSE_empirical_reg'].sum(),
              ranksums(data['SSE_empirical_reg'], data['SSE_DNN']))

    put_total = get_backtesting(
        total_data,
        start_date=datetime(2015, 1, 15),
        end_date=datetime(2019, 12, 30),
        cp=-1,
        rolling=False,
        TESTING_PERIOD=240 * 9,
        freq='M',
        info=['sigmoid', 0.0003, list([80, 80, 80, 80, 80, 80]), 500, True, 0.3],
        include_VKOSPI=False,
        include_macro=False
    )
    put_total.to_csv('data/DB/put_ensemble_last_gamma.csv', index=False, encoding='utf-8')
    print(delta_class_gain(put_total, 'empirical_reg', 'BS'))
    print(delta_class_gain(put_total, 'DNN', 'empirical_reg'))
    separated_BS = pd.pivot_table(put_total, values='SSE_DNN', columns=['delta_class'], index=['ut'], aggfunc=np.sum)
    separated_MV = pd.pivot_table(put_total, values='SSE_empirical_reg', columns=['delta_class'], index=['ut'],
                                  aggfunc=np.sum)
    print(1 - separated_BS / separated_MV)
    for idx, data in put_total.groupby('delta_class'):
        print(delta_class_gain(data, 'empirical_reg', 'BS'))
        print(idx, ranksums(data['SSE_empirical_reg'], data['SSE_BS']))

    for idx, data in put_total.groupby('delta_class'):
        print(delta_class_gain(data, 'DNN', 'empirical_reg'))
        print(idx, ranksums(data['SSE_empirical_reg'], data['SSE_DNN']))

    for idx, data in put_total.groupby(['delta_class', 'ut']):
        print(idx, 1 - data['SSE_DNN'].sum() / data['SSE_empirical_reg'].sum(),
              ranksums(data['SSE_empirical_reg'], data['SSE_DNN']))

# %% application appropriate
# call_df = pd.read_csv('data/DB/call_ensemble_last.csv')
# dfc = call_df.groupby('delta_class').apply(
#     lambda x: (LinearRegression().fit(x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'bs_'+GAMMA]], x[MKT_PRICE_CHG] - x['bs_' + DELTA] * x[KS200_CHG])).score(
#         x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'bs_'+GAMMA]], x[MKT_PRICE_CHG] - x['bs_' + DELTA] * x[KS200_CHG]))
#
# put_df = pd.read_csv('data/DB/put_ensemble_last.csv')
# dfp = put_df.groupby('delta_class').apply(
#     lambda x: (LinearRegression().fit(x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'bs_'+GAMMA]], x[MKT_PRICE_CHG] - x['bs_' + DELTA] * x[KS200_CHG])).score(
#         x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'bs_'+GAMMA]], x[MKT_PRICE_CHG] - x['bs_' + DELTA] * x[KS200_CHG]))
# dfo = pd.concat([dfc, dfp], axis=1)
# dfo.columns = ['call', 'put']
# dfo.plot()
# plt.show()
#
#
# # corr w input
# cp = 1
# unrefined_total_data = deepcopy(total_data)
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['delta_class'] = refined_total_data.apply(lambda x: delta_classifier2(x['bs_' + DELTA], x[CP]), axis=1)
# refined_total_data['reg_error'] = refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] * \
#                                   refined_total_data[KS200_CHG] - (
#                                       LinearRegression().fit(
#                                           refined_total_data[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA]],
#                                           refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] *
#                                           refined_total_data[KS200_CHG])).predict(
#     refined_total_data[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA]])
# dfc = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'reg_error']]).corr()['reg_error'])
#
# cp = -1
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['bs_' + DELTA + '_class'] = make_int(refined_total_data['bs_' + DELTA])
# refined_total_data['reg_error'] = refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] * \
#                                   refined_total_data[KS200_CHG] - (
#                                       LinearRegression().fit(
#                                           refined_total_data[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA]],
#                                           refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] *
#                                           refined_total_data[KS200_CHG])).predict(
#     refined_total_data[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA]])
# dfp = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'reg_error']]).corr()['reg_error'])
#
# dfp.sort_index(ascending=False, inplace=True)
# dfp.reset_index(drop=True, inplace=True)
# dfo = pd.concat([dfc, dfp], axis=1, ignore_index=True)
# dfo.columns = ['call', 'put']
# dfo.plot()
# plt.show()
#
# # error var
# cp = 1
# unrefined_total_data = deepcopy(total_data)
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['error'] = refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] * \
#                               refined_total_data[KS200_CHG]
# refined_total_data['bs_' + DELTA + '_class'] = make_int(refined_total_data['bs_' + DELTA])
# dfc_err_var = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'error']]).corr()['error'])
#
# cp = -1
# unrefined_total_data = deepcopy(total_data)
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['error'] = refined_total_data[MKT_PRICE_CHG] - refined_total_data['bs_' + DELTA] * \
#                               refined_total_data[KS200_CHG]
# refined_total_data['bs_' + DELTA + '_class'] = make_int(refined_total_data['bs_' + DELTA])
# dfp_err_var = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[[KS200_RET, TOW, 'bs_' + DELTA, 'bs_' + VEGA, 'error']]).corr()['error'])
#
# # trading volume
# cp = 1
# unrefined_total_data = deepcopy(total_data)
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['bs_' + DELTA + '_class'] = make_int(refined_total_data['bs_' + DELTA])
# dfc_vol = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[TRADING_VOLUME].mean()))
#
# cp = -1
# unrefined_total_data = deepcopy(total_data)
# refined_total_data = filtering(unrefined_total_data, cp, JD=True)
# refined_total_data['bs_' + DELTA + '_class'] = make_int(refined_total_data['bs_' + DELTA])
# dfp_vol = refined_total_data.groupby('bs_' + DELTA + '_class').apply(
#     lambda x: (x[TRADING_VOLUME].mean()))
#
# dfp_vol.sort_index(ascending=False, inplace=True)
# dfp_vol.reset_index(drop=True, inplace=True)
# dfo = pd.concat([dfc_vol, dfp_vol], axis=1, ignore_index=True)
# dfo.columns = ['call', 'put']
# # dfo.plot().hist(bins=10, alpha=0.5)
# # plt.hist(dfo.values, 10, alpha=0.5)
# plt.show()
#
# def cal_SSE(option):
#     return ((option[MKT_PRICE_CHG] - option['bs_' + DELTA] * option[KS200_CHG] - option['E_imp_chg_empirical_reg'] *
#              option['bs_' + VEGA]) ** 2).sum()
#
#
# def cal_SST(option):
#     return ((option[MKT_PRICE_CHG] - option['bs_' + DELTA] * option[KS200_CHG] - (
#             option[MKT_PRICE_CHG] - option['bs_' + DELTA] * option[KS200_CHG]).mean()) ** 2).sum()
#
#
# emp_call = pd.read_csv('data/DB/test_call_ensemble.csv')
# emp_call['bs_' + DELTA + '_class'] = make_int(emp_call['bs_' + DELTA])
# emp_call_df = emp_call.groupby('bs_' + DELTA + '_class').apply(lambda x: 1 - cal_SSE(x) / cal_SST(x))
#
# emp_put = pd.read_csv('data/DB/test_put_ensemble.csv')
# emp_put['bs_' + DELTA + '_class'] = make_int(emp_put['bs_' + DELTA])
# emp_put_df = emp_put.groupby('bs_' + DELTA + '_class').apply(lambda x: 1 - cal_SSE(x) / cal_SST(x))
#
# dfp.sort_index(ascending=False, inplace=True)
# dfp.reset_index(drop=True, inplace=True)
# dfo = pd.concat([dfc, dfp], axis=1, ignore_index=True)
# dfo.columns = ['call', 'put']
# dfo.plot()
# plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# x = call_df['bs_' + DELTA].values
# y = pd.to_datetime(call_df[TRADE_DATE], format='%Y-%m-%d').apply(lambda x: x.year * 10000 + x.month * 100 + x.day)
# x, y = np.meshgrid(x, y)
# z = call_df['BS_delta_hedge_error']
# ax.scatter(x, y, z)
# ax.plot_surface(x, y, z, rstride=4, cstride=4, alpha=0.4, cmap=cm.jet)
# plt.show()
