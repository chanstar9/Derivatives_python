# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 21
"""
from tqdm import tqdm
from columns import *
import pandas as pd
from trading_date.get_maturity import *


def index_imp_vol_preprocessor(file_name: str, id: int):
    """
    :param file_name:
    :param id:
    :return:
    """
    df = pd.read_excel('data/option/{}.xlsx'.format(file_name), skiprows=8, header=0)
    df[UNDERLYING] = df.apply(lambda x: underlying_identifier(x['Symbol Name'], id), axis=1)
    df[MAT] = df.apply(lambda x: mat_identifier(x['Symbol Name'], id), axis=1)
    df[MAT] = df.apply(lambda x: meetup_day(x[MAT].year, x[MAT].month, "Thursday", "2nd"), axis=1)
    df[K] = df.apply(lambda x: strike_identifier(x['Symbol Name']), axis=1)
    df[CP] = df.apply(lambda x: cp_identifier(x['Symbol Name']), axis=1)
    item_names = df['Item Name '].unique()
    item_len = len(item_names)
    df = df.drop(['Symbol', 'Symbol Name', 'Item Name ', 'Kind', 'Item', 'Frequency'], axis=1)
    melted_df = pd.DataFrame(columns=[UNDERLYING, MAT, K, CP, DATE])
    melted_df.set_index([UNDERLYING, MAT, K, CP, DATE], inplace=True)
    for idx, item in tqdm(enumerate(item_names)):
        item_df = pd.melt(df.iloc[idx::item_len, :], id_vars=[UNDERLYING, MAT, K, CP], var_name=DATE, value_name=item)
        item_df.set_index([UNDERLYING, MAT, K, CP, DATE], inplace=True)
        melted_df = melted_df.join(item_df, how='outer')
    melted_df.reset_index(inplace=True)
    melted_df.rename(
        columns={'종가(포인트)': CLOSE_P, '거래량(계약)': TRADING_VOLUME, '거래대금(천원)': TRANSACTION_AMT, '잔존일수(일)': TOW,
                 '내재변동성()': IMPLIED_VOL, DATE: TRADE_DATE}, inplace=True)
    melted_df.dropna(inplace=True)
    return melted_df


def underlying_identifier(symbol_name, id):
    if id == 0:
        if len(symbol_name.split()) == 5:
            return symbol_name.split()[0] + symbol_name.split()[1]
        else:
            return symbol_name.split()[0]
    elif id == 1:
        return symbol_name.split()[0]


def strike_identifier(symbol_name):
    if id == 0:
        return float(symbol_name.split()[-1])
    elif id == 1:
        return float(symbol_name.split()[-2][:-1].replace(',', ''))


def mat_identifier(symbol_name, id):
    if id == 0:
        if len(symbol_name.split()) == 5:
            return pd.to_datetime(symbol_name.split()[-2], format='%y%m')
        else:
            return pd.to_datetime(symbol_name.split()[-2], format='%Y%m')
    elif id == 1:
        if len(symbol_name.split()[-3]) == 4:
            return pd.to_datetime(symbol_name.split()[-3], format='%y%m')
        else:
            return pd.to_datetime(symbol_name.split()[-3], format='%Y%m')


def cp_identifier(symbol_name):
    if symbol_name.split()[-3] in ['콜옵션', 'C']:
        return 1
    else:
        return -1
