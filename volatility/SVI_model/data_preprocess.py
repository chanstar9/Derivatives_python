# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 08. 22
"""
import numpy as np
import pandas as pd

from trading_date.get_maturity import *

#
# maturities = ['202009', '202010', '202011', '202012', '202101', '202102', '202103', '202106', '202112', '202206',
#               '202212']
# s0 = 306.16
# r = 0.0079
# today = datetime(2020, 8, 21, 15, 30)
# option_data = pd.DataFrame(columns=["Expiration", "Strike", "Mid_Matrix"])
#
# for mat in maturities:
#     total = pd.read_excel("data/20200821_{}옵션종가.xlsx".format(mat), header=15)
#     tow = (meetup_day(int(mat[:4]), int(mat[4:]), "Thursday", "2nd") - today).days / 365
#     total["현재가"] = total.apply(
#         lambda x: x["현재가.1"] + s0 - x["행사가"] * np.exp(-r * tow) if x["행사가"] < s0 * np.exp(r * tow) else x["현재가"],
#         axis=1)
#     df = total[["현재가", "행사가"]]
#     df["Expiration"] = tow
#     df.rename(columns={"현재가": "Mid_Matrix", "행사가": "Strike"}, inplace=True)
#     option_data = pd.concat([option_data, df], axis=0)
#
# option_data.to_csv("data/KS200_option_data.csv", encoding="utf-8", index=False)

# hoga data
KS200 = 309
r = 0.0079
today = datetime(2020, 8, 24, 10, 30)
hoga_data1 = pd.ExcelFile('volatility/data/Hoga_2020-08-24.xlsx').parse('Sheet1', skiprows=1)
hoga_data2 = pd.ExcelFile('volatility/data/Hoga_2020-08-24.xlsx').parse('Sheet2', skiprows=1)

option = []
tow1 = (datetime(2020, 9, 10) - today).days / 365
for idx, row in hoga_data1.iterrows():
    option.append([tow1, hoga_data1.loc[idx]["행사가격"], hoga_data1.loc[idx]["콜매수호가"], hoga_data1.loc[idx]["콜미드"],
                   hoga_data1.loc[idx]["콜매도호가"]])

tow2 = (datetime(2020, 10, 8) - today).days / 365
for idx, row in hoga_data2.iterrows():
    option.append([tow2, hoga_data2.loc[idx]["행사가격"], hoga_data1.loc[idx]["콜매수호가"], hoga_data1.loc[idx]["콜미드"],
                   hoga_data1.loc[idx]["콜매도호가"]])

option_data = pd.DataFrame(option, columns=["Expiration", "Strike", "Bid_price", "Mid_Matrix", "Ask_price"])
option_data.to_csv("data/KS200_option_data.csv", encoding="utf-8", index=False)
