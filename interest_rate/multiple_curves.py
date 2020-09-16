# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 04
"""
import pandas as pd

swp_under = ["1L", "3L", "6L"]
swp_mat = ["1M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y",
           "25Y", "30Y", "40Y"]
swp_tenor = ["1W", "1M", "2M", "3M", "6M", "9M", "1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y",
             "15Y", "20Y", "25Y", "30Y", "40Y"]
swp_ticker = []
for i in swp_under:
    for j in swp_mat:
        for k in swp_tenor:
            swp_ticker.append("USDSB" + i + j + "F" + k + "=")

pd.DataFrame(swp_ticker).to_csv("data/interest_rate/LIBOR_FWD_SWP.csv", index=False)

IRS_under = ["1L", "3L", "6L"]
IRS_mat = ["1Y", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y"]
IRS_ticker = []
for i in IRS_under:
    for j in IRS_mat:
        IRS_ticker.append("USDSB" + i + j + "=TWEB")

pd.DataFrame(IRS_ticker).to_csv("data/interest_rate/LIBOR_SWP.csv", index=False)

Futures_under = ["EM", "ED"]
Futures_decade = [0, 1, 2, 3]
Futures_num = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Futures_month = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
Futures_ticker = []
for i in Futures_under:
    for j in Futures_decade:
        for k in Futures_num:
            for ll in Futures_month:
                Futures_ticker.append(i + ll + str(k) + "^" + str(j))

pd.DataFrame(Futures_ticker).to_csv("data/interest_rate/LIBOR_Futures.csv", index=False)
