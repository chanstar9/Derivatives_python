# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 17
"""
from copy import deepcopy
from interest_rate.quantlib_assistant_funcs import *


def discounting_curve(OIS_rate, date, calendar_country, settlementDays, OIS_tenors):
    _ois = deepcopy(OIS_rate)
    _ois.index = OIS_tenors
    _ois[QUOTE] = _ois.apply(lambda x: ql.SimpleQuote(x[date]), axis=1)
    FedFunds = ql.FedFunds()
    ois_helper = [ql.OISRateHelper(settlementDays, tenor, ql.QuoteHandle(_ois.loc[tenor][QUOTE]), FedFunds) for tenor in
                  OIS_tenors]
    day_count = ql.Actual360()
    D_curve = ql.PiecewiseCubicZero(0, calendar_country, ois_helper, day_count)  # dc 확인하기
    D_curve.enableExtrapolation()
    return D_curve


def get_multiple_curves(products, date, calendar_country, settlementDays, deposit_maturities, FRA_6L_maturities,
                        EURODOLLAR_maturities, swap_1L_tenors, swap_3L_tenors):
    # deposits
    _deposits = products.loc[DEPOSITS]
    _deposits.index = deposit_maturities
    _deposits[QUOTE] = _deposits.apply(lambda x: ql.SimpleQuote(x[date]), axis=1)
    depositHelpers = [
        ql.DepositRateHelper(ql.QuoteHandle(_deposits.loc[mat][QUOTE]), mat, settlementDays, calendar_country,
                             ql.ModifiedFollowing, False, ql.Actual360()) for mat in deposit_maturities]
    # fra
    _fra = products.loc[FRA].reset_index()
    _fra[TENOR] = _fra.apply(lambda x: int(x['index'].split('X')[1].replace('F=', '')) - int(
        x['index'].split('X')[0].replace('USD', '')), axis=1)
    _fra6L = dict(zip(FRA_6L_maturities,
                      _fra[_fra[TENOR] == 6].apply(lambda x: ql.SimpleQuote(x[date]), axis=1).to_list()))
    fra6L_Helpers = [
        ql.FraRateHelper(ql.QuoteHandle(_fra6L[mat]), mat[0], mat[1], settlementDays, calendar_country,
                         ql.ModifiedFollowing, False, ql.Actual360()) for mat in FRA_6L_maturities if mat[0] <= 12]
    # eurodollar
    _eurodollar = products.loc[EURODOLLAR_FUTURES]
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
    _swap = products.loc[SWAP_RATE].reset_index()
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

    return depoFuturesSwapCurve_1L, depoFuturesSwapCurve_3L, depoFRASwapCurve_6L
