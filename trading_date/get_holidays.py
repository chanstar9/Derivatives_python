# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2019. 11. 26
"""
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime
import holidays


# 증시 휴장
def get_holidays(county):
    nyse = mcal.get_calendar(county)
    holidays = nyse.holidays()
    return list(holidays.holidays)


today = datetime.now()
expiration = datetime(2019, 2, 13, 0, 0)
holidays = get_holidays('NYSE')  # NYSE Holidays

# count trading date
days_to_expiration = np.busday_count(today.date(), expiration.date(), holidays=holidays)
print(days_to_expiration)


def find_date():
    return


# Select country
uk_holidays = holidays.UnitedKingdom()

# Print all the holidays in UnitedKingdom in year 2018
for ptr in holidays.UnitedKingdom(years=2018).items():
    print(ptr)
