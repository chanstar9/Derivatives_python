# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_datareader.data as wb
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from arch import arch_model

# get data
codes = ['삼성전기', 'sk하이닉스', 'sk이노베이션', 'POSCO', '한국항공우주']
df = pd.DataFrame()
for code in codes:
    price = pd.read_excel('data/{}.xlsx'.format(code), index_col=0)
    price.columns = [code]
    term = 20
    ret = 100 * (np.log(price[code]) - np.log(price[code].shift(1))).dropna()

    # calculate vol
    sig_daily = ret ** 2

    # set train and test
    X_train = []
    y_train = []
    for i in range(len(ret) - term):
        X_train.append(np.array(ret[i:i + term]))
        y_train.append(np.array(sig_daily[i + term]))

    train_term = 2
    y_pred = []
    for i in range(len(X_train) - train_term):
        svm_model = SVR(gamma='scale', C=0.1, epsilon=0.2).fit(np.array(X_train[i:i + train_term]),
                                                               np.array(y_train[i:i + train_term]))
        y_pred.append(svm_model.predict(np.array([X_train[i + train_term]]))[0])
    df = pd.concat([df, pd.DataFrame(y_pred, columns=[code], index=price[term + train_term+1:].index)], axis=1)

    real_vol = sig_daily.rolling(5).mean()
    svm_vol = pd.DataFrame(y_pred, columns=['svm_vol']).rolling(5).mean()
    plt.plot(real_vol[term + train_term:].values)
    plt.plot(svm_vol)
    plt.show()

# for i in range(len(y_pred)):
#     print(sig_daily[20 + i] - y_pred[i])
