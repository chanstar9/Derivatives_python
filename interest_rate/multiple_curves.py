# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 17
"""
from datetime import timedelta

from trading_date.get_maturity import meetup_day
from interest_rate.quantlib_assistant_funcs import *

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

# product indexing
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
LIBOR_products[DEPOSITS] *= 0.01
LIBOR_products[FRA] *= 0.01
LIBOR_products[SWAP_RATE] *= 0.01
LIBOR_products[BASIS_SWAP_RATE] *= 0.0001

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

month_pair = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6, 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
EURODOLLAR_maturities = dict(zip(eurodollar.columns[1:], pd.DataFrame(eurodollar.columns[1:]).apply(
    lambda x: meetup_day(int('20' + x[0][-1] + x[0][-3]), month_pair[x[0][2]], "Wednesday", "3th", type='quantlib'),
    axis=1).dropna().values))

swap_1L_tenors = [ql.Period(2, ql.Years), ql.Period(3, ql.Years), ql.Period(4, ql.Years), ql.Period(5, ql.Years),
                  ql.Period(6, ql.Years), ql.Period(7, ql.Years), ql.Period(8, ql.Years), ql.Period(9, ql.Years),
                  ql.Period(10, ql.Years), ql.Period(12, ql.Years), ql.Period(15, ql.Years), ql.Period(20, ql.Years),
                  ql.Period(25, ql.Years), ql.Period(30, ql.Years), ql.Period(40, ql.Years)]
swap_3L_tenors = [ql.Period(1, ql.Years)] + swap_1L_tenors
# swap_6L_tenors = deepcopy(swap_1L_tenors)

# convention
calendar_country = ql.UnitedStates()
settlementDays = 2

# make curves every day
for date in dates:
    # setting time
    today = ql.Date(date.day, date.month, date.year)
    ql.Settings.instance().evaluationDate = today

    # select data
    LIBOR_product = LIBOR_products[LIBOR_products.index == date]
    LIBOR_product = LIBOR_product.dropna(axis='columns').T

    # deposits
    _deposits = LIBOR_product.loc[DEPOSITS]
    _deposits.index = deposit_maturities
    _deposits[QUOTE] = _deposits.apply(lambda x: ql.SimpleQuote(x[date]), axis=1)
    depositHelpers = [
        ql.DepositRateHelper(ql.QuoteHandle(_deposits.loc[mat][QUOTE]), mat, settlementDays, calendar_country,
                             ql.ModifiedFollowing, False, ql.Actual360()) for mat in deposit_maturities]
    # fra
    _fra = LIBOR_product.loc[FRA].reset_index()
    _fra[TENOR] = _fra.apply(lambda x: int(x['index'].split('X')[1].replace('F=', '')) - int(
        x['index'].split('X')[0].replace('USD', '')), axis=1)
    _fra3L = dict(zip(FRA_3L_maturities,
                      _fra[_fra[TENOR] == 3].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
    _fra6L = dict(zip(FRA_6L_maturities,
                      _fra[_fra[TENOR] == 6].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
    fra3L_Helpers = [
        ql.FraRateHelper(ql.QuoteHandle(_fra3L[mat]), mat[0], mat[1], settlementDays, calendar_country,
                         ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_3L_maturities if mat[0] < 12]
    fra6L_Helpers = [
        ql.FraRateHelper(ql.QuoteHandle(_fra6L[mat]), mat[0], mat[1], settlementDays, calendar_country,
                         ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_6L_maturities if mat[0] <= 12]
    # eurodollar
    _eurodollar = LIBOR_product.loc[EURODOLLAR_FUTURES]
    _eurodollar[MAT] = _eurodollar.apply(lambda x: EURODOLLAR_maturities[x.name], axis=1)
    _eurodollar[TOW] = _eurodollar.apply(
        lambda x: (datetime(x[MAT].year(), x[MAT].month(), x[MAT].dayOfMonth()) - date).days, axis=1)
    dayCounter = ql.Thirty360()
    convexityAdjustment = ql.QuoteHandle(ql.SimpleQuote(0))
    lengthInMonths = 1
    eurodollar1L_Helpers = [
        ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(_eurodollar.loc[ric][date])),
                             _eurodollar.loc[ric][MAT], lengthInMonths, calendar_country, ql.ModifiedFollowing,
                             True, dayCounter, convexityAdjustment) for ric in _eurodollar.index if
        ('EM' in ric) & (_eurodollar.loc[ric][TOW] >= 25) & (_eurodollar.loc[ric][TOW] <= 365 * 2)]
    lengthInMonths = 3
    eurodollar3L_Helpers = [
        ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(_eurodollar.loc[ric][date])),
                             _eurodollar.loc[ric][MAT], lengthInMonths, calendar_country, ql.ModifiedFollowing,
                             True, dayCounter, convexityAdjustment) for ric in _eurodollar.index if
        ('ED' in ric) & (_eurodollar.loc[ric][TOW] >= 25) & (_eurodollar.loc[ric][TOW] < 360 - 91)]
    # swap
    _swap = LIBOR_product.loc[SWAP_RATE].reset_index()
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
        ql.SwapRateHelper(ql.QuoteHandle(_swap1L[tenor]), tenor, calendar_country, fixedLegFrequency,
                          fixedLegAdjustment, fixedLegDayCounter, ql.Euribor1M()) for tenor in swap_1L_tenors]
    swap3L_Helpers = [
        ql.SwapRateHelper(ql.QuoteHandle(_swap3L[tenor]), tenor, calendar_country, fixedLegFrequency,
                          fixedLegAdjustment, fixedLegDayCounter, ql.Euribor3M()) for tenor in swap_3L_tenors]
    swap6L_Helpers = [
        ql.SwapRateHelper(ql.QuoteHandle(_swap6L[tenor]), tenor, calendar_country, fixedLegFrequency,
                          fixedLegAdjustment, fixedLegDayCounter, ql.Euribor6M()) for tenor in swap_1L_tenors]

    # term structure handles
    discountTermStructure = ql.RelinkableYieldTermStructureHandle()
    forecastTermStructure = ql.RelinkableYieldTermStructureHandle()

    # term-structure construction
    helpers_1L = depositHelpers[:5] + eurodollar1L_Helpers + swap1L_Helpers
    helpers_3L = depositHelpers[:5] + eurodollar3L_Helpers + swap3L_Helpers
    helpers_6L = depositHelpers[:5] + fra6L_Helpers + swap6L_Helpers

    day_count = ql.Actual360()
    depoFuturesSwapCurve_1L = ql.PiecewiseCubicZero(0, calendar_country, helpers_1L, day_count)  # dc 확인하기
    depoFuturesSwapCurve_1L.enableExtrapolation()
    depoFuturesSwapCurve_3L = ql.PiecewiseCubicZero(0, calendar_country, helpers_3L, day_count)  # dc 확인하기
    depoFuturesSwapCurve_3L.enableExtrapolation()
    depoFRASwapCurve_6L = ql.PiecewiseCubicZero(0, calendar_country, helpers_6L, day_count)  # dc 확인하기
    depoFRASwapCurve_6L.enableExtrapolation()

    # to DataFame
    ZC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='zero_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), under_name='1L')
    DC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='1L')
    FC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=30, under_name='1L')
    ZC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='zero_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), under_name='3L')
    DC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='3L')
    FC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=91, under_name='3L')
    ZC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='zero_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), under_name='6L')
    DC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='6L')
    FC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=180, under_name='6L')
    ZC = pd.concat([ZC1L, ZC3L['3L_zero_rate'], ZC6L['6L_zero_rate']], axis=1).set_index(DATE)
    DC = pd.concat([DC1L, DC3L['3L_discount'], DC6L['6L_discount']], axis=1).set_index(DATE)
    FC = pd.concat([FC1L, FC3L['3L_forward_rate'], FC6L['6L_forward_rate']], axis=1).set_index(DATE)

    # plot curves
    plot_curve(ZC, start_date=today, end_date=today + ql.Period(3, ql.Years), ylim=[-0.01, 0.02])
    plot_curve(DC, start_date=today, end_date=today + ql.Period(3, ql.Years), ylim=[-0.01, 0.02])
    plot_curve(FC, start_date=today, end_date=today + ql.Period(3, ql.Years), ylim=[-0.015, 0.03])

    plot_curve(ZC, start_date=today, end_date=ZC.index[-1], ylim=[ZC.values.min() - 0.005, ZC.values.max() + 0.005])
    plot_curve(DC, start_date=today, end_date=ZC.index[-1], ylim=[-0.01, 0.07])
    plot_curve(FC, start_date=today, end_date=ZC.index[-1], ylim=[FC.values.min() - 0.005, FC.values.max() + 0.005])

# def get_helpers(IR_product, date, deposits=False, fra=False, futures=False, swap=False):
#     IR_product = IR_product[IR_product.index == date]
#     IR_product = IR_product.dropna(axis='columns').T
#
#     if deposits:
#         _deposits = IR_product.loc[DEPOSITS]
#         _deposits.index = deposit_maturities
#         _deposits[QUOTE] = _deposits.apply(lambda x: ql.SimpleQuote(x[date]), axis=1)
#         depositHelpers = [
#             ql.DepositRateHelper(ql.QuoteHandle(_deposits.loc[mat][QUOTE]), mat, settlementDays, calendar_country,
#                                  ql.ModifiedFollowing, False, ql.Actual360()) for mat in deposit_maturities]
#     if fra:
#         _fra = IR_product.loc[FRA].reset_index()
#         _fra[TENOR] = _fra.apply(lambda x: int(x['index'].split('X')[1].replace('F=', '')) - int(
#             x['index'].split('X')[0].replace('USD', '')), axis=1)
#         _fra3L = dict(zip(FRA_3L_maturities,
#                           _fra[_fra[TENOR] == 3].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
#         _fra6L = dict(zip(FRA_6L_maturities,
#                           _fra[_fra[TENOR] == 6].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
#         fra3L_Helpers = [
#             ql.FraRateHelper(ql.QuoteHandle(_fra3L[mat]), mat[0], mat[1], settlementDays, calendar_country,
#                              ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_3L_maturities if mat[0] < 12]
#         fra6L_Helpers = [
#             ql.FraRateHelper(ql.QuoteHandle(_fra6L[mat]), mat[0], mat[1], settlementDays, calendar_country,
#                              ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_6L_maturities]
#     if futures:
#         _eurodollar = IR_product.loc[EURODOLLAR_FUTURES]
#         _eurodollar[MAT] = _eurodollar.apply(lambda x: EURODOLLAR_maturities[x.name], axis=1)
#         _eurodollar[TOW] = _eurodollar.apply(
#             lambda x: (datetime(x[MAT].year(), x[MAT].month(), x[MAT].dayOfMonth()) - date).days, axis=1)
#         dayCounter = ql.Thirty360()
#         convexityAdjustment = ql.QuoteHandle(ql.SimpleQuote(0))
#         lengthInMonths = 1
#         eurodollar1L_Helpers = [
#             ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(_eurodollar.loc[ric][date])),
#                                  _eurodollar.loc[ric][MAT], lengthInMonths, calendar_country, ql.ModifiedFollowing,
#                                  True, dayCounter, convexityAdjustment) for ric in _eurodollar.index if
#             ('EM' in ric) & (_eurodollar.loc[ric][TOW] >= 25) & (_eurodollar.loc[ric][TOW] <= 365 * 2)]
#         lengthInMonths = 3
#         eurodollar3L_Helpers = [
#             ql.FuturesRateHelper(ql.QuoteHandle(ql.SimpleQuote(_eurodollar.loc[ric][date])),
#                                  _eurodollar.loc[ric][MAT], lengthInMonths, calendar_country, ql.ModifiedFollowing,
#                                  True, dayCounter, convexityAdjustment) for ric in _eurodollar.index if
#             ('ED' in ric) & (_eurodollar.loc[ric][TOW] >= 25) & (_eurodollar.loc[ric][TOW] <= 365)]
#     if swap:
#         _swap = IR_product.loc[SWAP_RATE].reset_index()
#         _swap1L = dict(zip(swap_1L_tenors,
#                            _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '1L' in x['index'] else None,
#                                        axis=1).dropna()))
#         _swap3L = dict(zip(swap_3L_tenors,
#                            _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '3L' in x['index'] else None,
#                                        axis=1).dropna()))
#         _swap6L = dict(zip(swap_1L_tenors,
#                            _swap.apply(lambda x: ql.SimpleQuote(x[date]) if '6L' in x['index'] else None,
#                                        axis=1).dropna()))
#         fixedLegFrequency = ql.Semiannual
#         fixedLegAdjustment = ql.Unadjusted
#         fixedLegDayCounter = ql.Actual365Fixed()
#         swap1L_Helpers = [
#             ql.SwapRateHelper(ql.QuoteHandle(_swap1L[tenor]), tenor, calendar_country, fixedLegFrequency,
#                               fixedLegAdjustment, fixedLegDayCounter, ql.Euribor1M()) for tenor in swap_1L_tenors]
#         swap3L_Helpers = [
#             ql.SwapRateHelper(ql.QuoteHandle(_swap3L[tenor]), tenor, calendar_country, fixedLegFrequency,
#                               fixedLegAdjustment, fixedLegDayCounter, ql.Euribor3M()) for tenor in swap_3L_tenors]
#         swap6L_Helpers = [
#             ql.SwapRateHelper(ql.QuoteHandle(_swap6L[tenor]), tenor, calendar_country, fixedLegFrequency,
#                               fixedLegAdjustment, fixedLegDayCounter, ql.Euribor6M()) for tenor in swap_1L_tenors]
#     return
