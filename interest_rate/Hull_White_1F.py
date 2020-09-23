# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 09. 21
"""
import QuantLib as ql
from collections import namedtuple

today = ql.Date(15, ql.February, 2002)
settlement = ql.Date(19, ql.February, 2002)
ql.Settings.instance().evaluationDate = today
term_structure = ql.YieldTermStructureHandle(ql.FlatForward(settlement, 0.04875825, ql.Actual365Fixed()))
index = ql.Euribor1Y(term_structure)

