# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 17
"""

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import QuantLib as ql

from columns import *


def to_datetime(d):
    return datetime(d.year(), d.month(), d.dayOfMonth())


def format_rate(r):
    return '%.4f %%' % (r.rate() * 100.0)


def curve_to_DataFrame(curves, today, curve_type, compound, daycounter=None, forward_tenor=0, under_name=None):
    _dates = [today + ql.Period(i, ql.Days) for i in range(0, curves.dates()[-1] - today)]
    valid_dates = [d for d in _dates if d >= curves.referenceDate()]
    if curve_type == 'zero_curve':
        if not daycounter:
            ValueError("setting daycount convention")
        rates = [curves.zeroRate(d, daycounter, compound).rate() for d in valid_dates]
        return pd.DataFrame(np.array([valid_dates, rates]).T, columns=[DATE, under_name + '_zero_rate'])
    if curve_type == 'discount_curve':
        rates = [curves.discount(d) for d in valid_dates]
        return pd.DataFrame(np.array([valid_dates, rates]).T, columns=[DATE, under_name + '_discount'])
    if curve_type == 'forward_curve':
        if forward_tenor == 0:
            ValueError("forward_tenor must be greater than 0")
        rates = [curves.forwardRate(d, d + forward_tenor, daycounter, compound).rate() for d in valid_dates]
        return pd.DataFrame(np.array([valid_dates, rates]).T, columns=[DATE, under_name + '_forward_rate'])
    else:
        ValueError("check curve type")


def plot_curve(curves, start_date: ql.Date, end_date: ql.Date):
    _dates = [d for d in curves.index if (start_date <= d) & (d <= end_date)]
    fig, ax = plt.subplots()
    plt.rc('lines', linewidth=3)
    for idx, value in curves.iteritems():
        ax.plot_date([to_datetime(d) for d in _dates], value.loc[_dates].values, '-')
    ax.set_xlim(to_datetime(min(_dates)), to_datetime(max(_dates)))
    ax.xaxis.set_major_locator(MonthLocator(bymonth=[12]))
    ax.xaxis.set_major_formatter(DateFormatter("%b '%y"))
    ax.set_ylim(curves.values.min() - 0.005, curves.values.max() + 0.005)
    ax.autoscale_view()
    ax.xaxis.grid(True, 'major')
    ax.xaxis.grid(False, 'minor')
    ax.legend(curves.columns)
    ax.set_title(start_date)
    fig.autofmt_xdate()
    plt.show()
