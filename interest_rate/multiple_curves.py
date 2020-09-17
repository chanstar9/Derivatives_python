# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 17
"""
from copy import deepcopy
from datetime import datetime, timedelta

import pandas as pd
import QuantLib as ql

from columns import *

# data preprocess
LIBOR_TOTAL = pd.ExcelFile('data/interest_rate/LIBOR_TOTAL.xlsx')
deposits = LIBOR_TOTAL.parse(DEPOSITS)
fra = LIBOR_TOTAL.parse(FRA)
eurodollar = LIBOR_TOTAL.parse(EURODOLLAR_FUTURES)
swap = LIBOR_TOTAL.parse(SWAP_RATE)
basis = LIBOR_TOTAL.parse(BASIS_SWAP_RATE)
LIBOR_products = pd.merge(deposits, fra, on='Timestamp', how='outer')
LIBOR_products = pd.merge(LIBOR_products, eurodollar, on='Timestamp', how='outer')
LIBOR_products = pd.merge(LIBOR_products, swap, on='Timestamp', how='outer')
LIBOR_products = pd.merge(LIBOR_products, basis, on='Timestamp', how='outer')
LIBOR_products.set_index('Timestamp', inplace=True)
LIBOR_products.sort_index(inplace=True)
multi_index = []
IR_products = [DEPOSITS, FRA, EURODOLLAR_FUTURES, SWAP_RATE, BASIS_SWAP_RATE]
pair = dict(zip(IR_products, [len(deposits.columns) - 1, len(fra.columns) - 1, len(eurodollar.columns) - 1,
                              len(swap.columns) - 1, len(basis.columns) - 1]))
for i in IR_products:
    lst = [i] * pair[i]
    multi_index += lst
LIBOR_products.columns = [multi_index, LIBOR_products.columns]
# LIBOR_product.fillna(method='ffill', inplace=True)

# filtering date
start_date = datetime(2010, 11, 15)
end_date = datetime(2020, 6, 30)
dates = []
for i in range((end_date - start_date).days + 1):
    dates.append(start_date + timedelta(days=i))
LIBOR_products = LIBOR_products[(LIBOR_products.index >= start_date) & (LIBOR_products.index <= end_date)]

# define maturities/tenors
deposit_maturities = [ql.Period(1, ql.Days), ql.Period(3, ql.Days), ql.Period(7, ql.Days),
                      ql.Period(14, ql.Days), ql.Period(21, ql.Days), ql.Period(31, ql.Days), ql.Period(61, ql.Days),
                      ql.Period(91, ql.Days), ql.Period(123, ql.Days), ql.Period(153, ql.Days), ql.Period(181, ql.Days),
                      ql.Period(213, ql.Days), ql.Period(242, ql.Days), ql.Period(273, ql.Days),
                      ql.Period(304, ql.Days), ql.Period(334, ql.Days), ql.Period(367, ql.Days),
                      ql.Period(458, ql.Days), ql.Period(546, ql.Days), ql.Period(640, ql.Days),
                      ql.Period(731, ql.Days)]
FRA_3L_maturities = [(1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (12, 15), (15, 18),
                     (18, 21)]
FRA_6L_maturities = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12), (9, 15), (12, 18), (18, 24)]
swap_1L_tenors = [ql.Period(2, ql.Years), ql.Period(3, ql.Years), ql.Period(4, ql.Years), ql.Period(5, ql.Years),
                  ql.Period(6, ql.Years), ql.Period(7, ql.Years), ql.Period(8, ql.Years), ql.Period(9, ql.Years),
                  ql.Period(10, ql.Years), ql.Period(12, ql.Years), ql.Period(15, ql.Years), ql.Period(20, ql.Years),
                  ql.Period(25, ql.Years), ql.Period(30, ql.Years), ql.Period(40, ql.Years)]
swap_3L_tenors = [ql.Period(1, ql.Years)] + swap_1L_tenors
# swap_6L_tenors = deepcopy(swap_1L_tenors)

# convention
calendar = ql.UnitedStates()
settlementDays = 2

# make curves every day
for date in dates:
    today = ql.Date(date.day, date.month, date.year)
    LIBOR_product = LIBOR_products[LIBOR_products.index == date]
    LIBOR_product = LIBOR_product.dropna(axis='columns').T
    for product in IR_products:
        # convert quote object
        if product == DEPOSITS:
            _deposits = LIBOR_product.loc[product]
            _deposits.index = deposit_maturities
            _deposits[QUOTE] = _deposits.apply(lambda x: ql.SimpleQuote(x[date]), axis=1)
            depositHelpers = [
                ql.DepositRateHelper(ql.QuoteHandle(_deposits.loc[mat][QUOTE]), mat, settlementDays, calendar,
                                     ql.ModifiedFollowing, False, ql.Actual360()) for mat in deposit_maturities]
        if product == FRA:
            _fra = LIBOR_product.loc[product].reset_index()
            _fra[TENOR] = _fra.apply(lambda x: int(x['index'].split('X')[1].replace('F=', '')) - int(
                x['index'].split('X')[0].replace('USD', '')), axis=1)
            _fra3L = dict(zip(FRA_3L_maturities,
                              _fra[_fra[TENOR] == 3].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
            _fra6L = dict(zip(FRA_6L_maturities,
                              _fra[_fra[TENOR] == 6].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
            fra3L_Helpers = [
                ql.FraRateHelper(ql.QuoteHandle(_fra3L[mat]), mat[0], mat[1], settlementDays, calendar,
                                 ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_3L_maturities]
            fra6L_Helpers = [
                ql.FraRateHelper(ql.QuoteHandle(_fra6L[mat]), mat[0], mat[1], settlementDays, calendar,
                                 ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_6L_maturities]
        if product == EURODOLLAR_FUTURES:
            _eurodollar = LIBOR_product.loc[product].reset_index()
            # eurodollar1L =
            dayCounter = ql.Thirty360()


        if product == SWAP_RATE:
            _swap = LIBOR_product.loc[product].reset_index()
            _swap1L = dict(zip(swap_1L_tenors,
                               _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '1L' in x['index'] else None,
                                           axis=1).dropna()))
            _swap3L = dict(zip(swap_3L_tenors,
                               _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '3L' in x['index'] else None,
                                           axis=1).dropna()))
            _swap6L = dict(zip(swap_1L_tenors,
                               _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '6L' in x['index'] else None,
                                           axis=1).dropna()))
            fixedLegFrequency = ql.Semiannual
            fixedLegAdjustment = ql.Unadjusted
            fixedLegDayCounter = ql.Actual365Fixed()
            swap1L_Helpers = [
                ql.SwapRateHelper(ql.QuoteHandle(_swap1L[tenor]), tenor, calendar, fixedLegFrequency,
                                  fixedLegAdjustment, fixedLegDayCounter, ql.Euribor1M()) for tenor in swap_1L_tenors]
            swap1L_Helpers = [
                ql.SwapRateHelper(ql.QuoteHandle(_swap3L[tenor]), tenor, calendar, fixedLegFrequency,
                                  fixedLegAdjustment, fixedLegDayCounter, ql.Euribor3M()) for tenor in swap_3L_tenors]
            swap1L_Helpers = [
                ql.SwapRateHelper(ql.QuoteHandle(_swap6L[tenor]), tenor, calendar, fixedLegFrequency,
                                  fixedLegAdjustment, fixedLegDayCounter, ql.Euribor6M()) for tenor in swap_1L_tenors]

