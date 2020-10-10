# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 17
"""
from datetime import timedelta

from trading_date.get_maturity import meetup_day
from interest_rate.generate_multiple_curves import *

# data preprocess
LIBOR_TOTAL = pd.ExcelFile('data/interest_rate/LIBOR_TOTAL.xlsx')
ois = LIBOR_TOTAL.parse(OIS).set_index('Timestamp') * 0.01
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
OIS_tenors = [ql.Period(1, ql.Months), ql.Period(2, ql.Months), ql.Period(3, ql.Months), ql.Period(4, ql.Months),
              ql.Period(5, ql.Months), ql.Period(6, ql.Months), ql.Period(7, ql.Months), ql.Period(8, ql.Months),
              ql.Period(9, ql.Months), ql.Period(10, ql.Months), ql.Period(11, ql.Months), ql.Period(1, ql.Years),
              ql.Period(15, ql.Months), ql.Period(18, ql.Months), ql.Period(21, ql.Months), ql.Period(2, ql.Years),
              ql.Period(3, ql.Years), ql.Period(4, ql.Years), ql.Period(5, ql.Years), ql.Period(6, ql.Years),
              ql.Period(7, ql.Years), ql.Period(8, ql.Years), ql.Period(9, ql.Years), ql.Period(10, ql.Years),
              ql.Period(12, ql.Years), ql.Period(15, ql.Years), ql.Period(20, ql.Years), ql.Period(25, ql.Years),
              ql.Period(30, ql.Years), ql.Period(40, ql.Years)]
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
    lambda x: meetup_day(int('20' + x[0][-1] + x[0][-3]), month_pair[x[0][2]], "Wednesday", "3th",
                         data_type='quantlib'),
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
    OIS_rate = ois[ois.index == date].T
    LIBOR_product = LIBOR_products[LIBOR_products.index == date]
    LIBOR_product = LIBOR_product.dropna(axis='columns').T

    # get multiple curves
    D_curve = discounting_curve(OIS_rate, date, calendar_country, settlementDays, OIS_tenors)
    depoFuturesSwapCurve_1L, depoFuturesSwapCurve_3L, depoFRASwapCurve_6L = \
        get_multiple_curves(LIBOR_product, date, calendar_country, settlementDays, deposit_maturities,
                            FRA_6L_maturities, EURODOLLAR_maturities, swap_1L_tenors, swap_3L_tenors)

    # to DataFame
    OIS_curve = curve_to_DataFrame(D_curve, today, curve_type='zero_curve', compound=ql.Continuous,
                                   daycounter=ql.Actual360(), under_name='OIS')
    # ZC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='zero_curve', compound=ql.Continuous,
    #                           daycounter=ql.Actual360(), under_name='1L')
    ZC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='zero_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), under_name='3L')
    ZC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='zero_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), under_name='6L')
    ZC = pd.concat([OIS_curve, ZC3L['3L_zero_rate'], ZC6L['6L_zero_rate']], axis=1).set_index(DATE)
    OIS_DISC = curve_to_DataFrame(D_curve, today, curve_type='discount_curve', compound=ql.Continuous, under_name='OIS')
    DC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='1L')
    DC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='3L')
    DC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='discount_curve', compound=ql.Continuous,
                              under_name='6L')
    DC = pd.concat([DC1L, DC3L['3L_discount'], DC6L['6L_discount']], axis=1).set_index(DATE)
    FC1L = curve_to_DataFrame(depoFuturesSwapCurve_1L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=30, under_name='1L')
    FC3L = curve_to_DataFrame(depoFuturesSwapCurve_3L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=91, under_name='3L')
    FC6L = curve_to_DataFrame(depoFRASwapCurve_6L, today, curve_type='forward_curve', compound=ql.Continuous,
                              daycounter=ql.Actual360(), forward_tenor=180, under_name='6L')
    FC = pd.concat([FC1L, FC3L['3L_forward_rate'], FC6L['6L_forward_rate']], axis=1).set_index(DATE)

    # plot curves
    plot_curve(ZC, start_date=today, end_date=today + ql.Period(3, ql.Years))
    # plot_curve(DC, start_date=today, end_date=today + ql.Period(3, ql.Years))
    plot_curve(FC, start_date=today, end_date=today + ql.Period(3, ql.Years))

    plot_curve(ZC, start_date=today, end_date=ZC.index[-1])
    # plot_curve(DC, start_date=today, end_date=ZC.index[-1])
    plot_curve(FC, start_date=today, end_date=ZC.index[-1])

    # save
    tmp = [True if i in depoFuturesSwapCurve_3L.dates() else False for i in DC3L['date'].values]
    aa = DC3L[tmp]
    aa['date'] = aa['date'].apply(lambda x: to_datetime(x))
    aa.to_csv('data/{}_3L.csv'.format(date))
