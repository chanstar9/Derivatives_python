# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 12
"""
import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

today = datetime.datetime(2019, 11, 4)
law = pd.read_excel('data/raw_data_of_zero_curve.xlsx')


# calculate TTM
def cal_TTM(x, date):
    if isinstance(x, datetime.datetime):
        return (x - date).days
    else:
        if 'W' in x:
            return int(x[:-1]) * 7
        if 'M' in x:
            return int(x[:-1]) * 30
        if 'Y' in x:
            return int(x[:-1]) * 360


law['days'] = law['Term'].map(lambda x: cal_TTM(x, today))

# calculate zero rate using MMD
MMD = law[law['Instrument'] == 'MMD']
MMD['Z(0,T)'] = 1 / (1 + MMD['Rate/Price'] * 0.01 * MMD['days'] / 360)
MMD['zero_rate'] = -np.log(MMD['Z(0,T)']) / MMD['days'] * 360

# interpolate zero rate curve
zero_rate_dates = []
for i in MMD['days']:
    zero_rate_dates.append(today + datetime.timedelta(days=int(i)))
zero_rate = pd.DataFrame(MMD['zero_rate'].values, index=zero_rate_dates, columns=['zero_rate'])
zero_rate = zero_rate.resample('D').last()
zero_rate.interpolate(method='piecewise_polynomial', inplace=True)
zero_rate['days'] = (zero_rate.index - today).days
zero_rate['Z(0,T)'] = 1 / (1 + zero_rate['days'] / 360 * zero_rate['zero_rate'])

# calculate zero rate using Futures
Futures = law[law['Instrument'] == 'Futures'].iloc[:-1]
Futures['T2'] = Futures['days'] + 91
Futures['Futures_rate'] = 1 - Futures['Rate/Price'] / 100
Futures['Convexity'] = 0.005 ** 2 * (Futures['T2'] / 360) * (Futures['days'] / 360) / 2
Futures['forward_rate'] = Futures['Futures_rate'] - Futures['Convexity']
z1 = []
for i in Futures['days']:
    z1.append(zero_rate[zero_rate['days'] == i]['Z(0,T)'].values[0])
Futures['Z(0,T1)'] = z1
Futures['Z(0,T2)'] = Futures['Z(0,T1)'] / (1 + ((Futures['T2'] - Futures['days']) / 360) * Futures['forward_rate'])
Futures['zero_rate'] = -np.log(Futures['Z(0,T2)']) / Futures['T2'] * 360

# append to zero rate curve
df = Futures[['Term', 'zero_rate', 'T2', 'Z(0,T2)']]
df['Term'] = df['Term'].apply(lambda x: x + relativedelta(days=30))
zero_rate = zero_rate[zero_rate.index < df['Term'].iloc[0]]
df.rename(columns={'T2': 'days', 'Z(0,T2)': 'Z(0,T)'}, inplace=True)
df.set_index('Term', inplace=True)
zero_rate = pd.concat([zero_rate, df], axis=0)
zero_rate = zero_rate.resample('D').last()
zero_rate[['zero_rate', 'days']] = zero_rate[['zero_rate', 'days']].interpolate(method='piecewise_polynomial')
zero_rate['Z(0,T)'] = np.exp(-zero_rate['days'] / 360 * zero_rate['zero_rate'])
zero_rate['days'] = zero_rate['days'].apply(lambda x: int(x))

# calculate zero rate using swap
Swap = law[law['Instrument'] == 'Swap']
d = Swap['days'].iloc[-1]
while d != 180:
    d -= 180
    if d not in Swap['days'].values:
        Swap = pd.concat([Swap, pd.DataFrame(['Swap', '{}Y'.format(d / 360), np.NaN, d], index=Swap.columns).T], axis=0)
Swap.sort_values('days', inplace=True)
Swap.reset_index(inplace=True, drop=True)
Swap['Rate/Price'] = Swap['Rate/Price'].interpolate(method='piecewise_polynomial')
Swap['Z(0,T)'] = zero_rate[zero_rate['days'] == Swap['days'].iloc[0]]['Z(0,T)'].values[0]
for idx in Swap.index[1:]:
    Swap['Z(0,T)'].iloc[idx] = (1 - Swap.loc[idx]['Rate/Price'] / 2 * 0.01 * np.sum(
        Swap['Z(0,T)'].loc[:idx - 1])) / (1 + Swap['Rate/Price'].loc[idx] / 2 * 0.01)
Swap['zero_rate'] = -np.log(Swap['Z(0,T)']) / Swap['days'] * 360

# append to zero rate curve
df = Swap[['zero_rate', 'days', 'Z(0,T)']]
t = []
for i in df['days']:
    t.append(today + relativedelta(days=int(i)))
df.index = t
zero_rate = zero_rate[zero_rate.index < df.index[0]]
zero_rate = pd.concat([zero_rate, df], axis=0)
zero_rate = zero_rate.resample('D').last()
zero_rate[['zero_rate', 'days']] = zero_rate[['zero_rate', 'days']].interpolate(method='piecewise_polynomial')
zero_rate['Z(0,T)'] = np.exp(-zero_rate['days'] / 360 * zero_rate['zero_rate'])
zero_rate['days'] = zero_rate['days'].apply(lambda x: int(x))

# graph
plt.plot(zero_rate['zero_rate'])
plt.savefig('picture/zero_rate_curve.png')
plt.show()

# save data
zero_rate.to_csv('data/zero_rate_curve.csv')
