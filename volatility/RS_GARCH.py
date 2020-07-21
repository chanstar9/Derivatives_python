# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date:
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# NBER recessions
import pandas_datareader.data as wb
from datetime import datetime


codes = ['삼성전기', 'sk하이닉스', 'sk이노베이션', 'POSCO', '한국항공우주']
for code in codes:
    price = pd.read_excel('data/{}.xlsx'.format(code), index_col=0)
    price.columns = [code]
    ret_data = 100 * (np.log(price[code]) - np.log(price[code].shift(1)))
    ret_data.dropna(inplace=True)
    ret_data.to_excel('data/{}_1.xlsx'.format(code))


# sigma_data = ret_data.rolling(5).std()
# sigma_data.dropna(inplace=True)
# Plot the data
sigma_data = np.sqrt(252) * (ret_data ** 2)
sigma_data.plot(title='daily vol', figsize=(12, 3))

# Fit the model
mod_hamilton = sm.tsa.MarkovAutoregression(sigma_data, k_regimes=2, trend='nc', order=1, switching_variance=True)
res_hamilton = mod_hamilton.fit()

# using R
