# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 09. 15
"""
# base information
DATE = 'date'
TRADE_DATE = 'trade_date'
CLOSE_P = 'close_price'
TRADING_VOLUME = 'trading_volume'
TRANSACTION_AMT = 'transaction_amount'
CALL = 'call'
PUT = 'put'

# option information
MKT_PRICE_CHG = 'mkt_price_chg'
VALUE = 'value'
K = 'strike_price'
TOW = 'time_to_maturity'
MAT = 'maturity'
LONG_SHORT = 'long_short'
CP = 'call_put'
MONEYNESS = 'moneyness'

# greek
DELTA = 'delta'
VEGA = 'vega'
GAMMA = 'gamma'
THETA = 'theta'
RHO = 'rho'

# kind of volatility
HIST_VOL = 'historical_volatility'
IMPLIED_VOL = 'implied_volatility'
LOCAL_VOL = 'local_volatility'

# underlying information
UNDERLYING = 'underlying'
S = 'underlying_price'
KS200 = 'kospi200'
KS200_CHG = 'kospi200_change'
KS200_RET = 'kospi200_ret'
KS200_F = 'kospi200_futures'
KS200_F_CHG = 'kospi200_futures_change'
KS200_F_RET = 'kospi200_futures_ret'
VKOSPI = 'vkospi'
KS200_DIV = 'kospi200_dividend_rate'
Q = 'dividend_rate'
VOL = 'volatility'
SPX = 'S&P500'
EUROSTOXX = 'eurostoxx'
INDICES = [KS200, SPX, EUROSTOXX]

# interest rate
R = 'interest_rate'
CD91 = 'cd91_rate'
KO_ONE_YEAR_TREASURY_RATE = 'korea_1year_treasury_rate'

# input information
COL = [S, K, TOW, R, Q, VOL, CLOSE_P, LONG_SHORT, CP]
