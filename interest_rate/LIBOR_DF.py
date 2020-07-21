# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 03. 03
"""
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from trading_date.get_maturity import *


def get_LIBOR_DF(today, ):
    return


today = datetime(2012, 11, 19)
df = pd.read_excel('data/interest_rate/Eurodollar_futures.xlsx', skiprows=3)
df['Maturity'] = df['Maturity'].apply(lambda x: pd.to_datetime(meetup_day(x.year, x.month, "Wednesday", "3nd")))
df['TTM'] = (df['Maturity'] - today).apply(lambda x: x.days) / 360
df['adj_LIBOR'] = df['Yield'] - (0.01 ** 2) * ((df['TTM'] ** 2) / 2 + df['TTM'] / 8)
df['LIBOR_DF'] = 1
for i in range(len(df) - 1):
    df['LIBOR_DF'].iloc[i + 1] = df['LIBOR_DF'].iloc[i] / (
            1 + df['adj_LIBOR'].iloc[i] * (df['TTM'].iloc[i + 1] - df['TTM'].iloc[i]))
swap_start_date = df['Maturity'].apply(lambda x: np.busday_offset(x.date(), -2, roll='forward'))
df.set_index('Maturity', inplace=True)
DF = df['LIBOR_DF'].resample('D').last()
DF.interpolate(method='linear', inplace=True)
answer = DF.loc[swap_start_date]
answer.iloc[0] = 1
answer.to_excel('data/interest_rate/LIBOR_DF.xlsx')




# calculate TTM
def cal_TTM(x, date):
    if isinstance(x, datetime):
        return (x - date).days
    else:
        if 'd' == x.split()[-1][0]:
            return int(x.split()[0])
        if 'w' == x.split()[-1][0]:
            return int(x.split()[0]) * 7
        if 'm' == x.split()[-1][0]:
            return int(x.split()[0]) * 30
        if 'y' == x.split()[-1][0]:
            return int(x.split()[0]) * 360


ois = pd.read_excel('data/interest_rate/OIS_rate.xlsx', skiprows=3)
ois['TTM'] = ois['Maturity'].apply(lambda x: cal_TTM(x, today)) / 360
ois['Maturity'] = ois['TTM'].apply(lambda x: today + timedelta(days=x * 360))
ois['OIS_DF'] = np.exp(-ois['Yield'] * ois['TTM'])
ois.set_index('Maturity', inplace=True)
DF = ois['OIS_DF'].resample('D').last()
DF.interpolate(method='linear', inplace=True)
answer = DF.loc[swap_start_date]
answer.iloc[0] = 1
answer.to_excel('data/interest_rate/OIS_DF.xlsx')
