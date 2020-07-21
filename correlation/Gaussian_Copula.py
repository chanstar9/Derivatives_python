# -*- coding: utf-8 -*-
"""
:Author: Chankyu Choi
:Date: 2020. 04. 25
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

# 1
loan_amt = [8000000] * 100 + [20000000] * 20
default_prob = 0.012
LGD = 0.5
time_horizon = 1


# 1(a)
def independent_default_loss(beta=False):
    df = pd.DataFrame(loan_amt, columns=['principle'])
    df['1-year_default_prob'] = default_prob
    df['1-year_survival_prob'] = 1 - default_prob
    df['annualized_default_rate'] = -np.log(df['1-year_survival_prob'])
    df['uncorr_Z'] = np.random.normal(0, 1, len(df))
    df['default_time'] = -np.log(1 - norm.cdf(df['uncorr_Z'])) / df['annualized_default_rate']
    if not beta:
        df['default_amt'] = df.apply(lambda x: x['principle'] * LGD if x['default_time'] <= time_horizon else 0, axis=1)
    else:
        df['default_amt'] = df.apply(
            lambda x: x['principle'] * np.random.beta(2, 2) if x['default_time'] <= time_horizon else 0, axis=1)
    return df


scenario1 = []
for _ in range(5000):
    scenario1.append(independent_default_loss()['default_amt'].sum())
pd.Series(scenario1).hist(bins=20)
plt.show()

# 1(b)
copula_correlation = 0.3


def single_factor_gaussian_copula_loss(copula_correlation, loan_amt, default_prob, beta=False):
    common_factor = norm.ppf(np.random.uniform(0, 1))
    df = pd.DataFrame(loan_amt, columns=['principle'])
    df['1-year_default_prob'] = default_prob
    df['1-year_survival_prob'] = 1 - default_prob
    df['annualized_default_rate'] = -np.log(df['1-year_survival_prob'])
    df['idiosyncratic_factor'] = norm.ppf(np.random.uniform(0, 1, len(df)))
    df['corr_Z'] = copula_correlation ** (1 / 2) * common_factor + (1 - copula_correlation) ** (1 / 2) * df[
        'idiosyncratic_factor']
    df['default_time'] = -np.log(1 - norm.cdf(df['corr_Z'])) / df['annualized_default_rate']
    if not beta:
        df['default_amt'] = df.apply(lambda x: x['principle'] * LGD if x['default_time'] <= time_horizon else 0, axis=1)
    else:
        df['default_amt'] = df.apply(
            lambda x: x['principle'] * np.random.beta(2, 2) if x['default_time'] <= time_horizon else 0, axis=1)
    return df


scenario2 = []
for _ in range(5000):
    scenario2.append(
        single_factor_gaussian_copula_loss(copula_correlation, loan_amt, default_prob)['default_amt'].sum())
pd.Series(scenario2).hist(bins=20)
plt.show()


# 1(c)


# 1(d)
def default_rate(copula_correlation, simulate_num):
    lst = []
    for _ in range(simulate_num):
        df = single_factor_gaussian_copula_loss(copula_correlation, [1] * 2000, default_prob)
        lst.append(df['default_amt'].sum() / df['principle'].sum())
    return np.array(lst).std()


print(default_rate(copula_correlation, 5000))

# 1(e)
from scipy.optimize import fsolve


def find_copula_corr(copula_correlation, simulate_num):
    return default_rate(copula_correlation, simulate_num) - 0.2


result = fsolve(find_copula_corr, 0.3, 5000)

# 2
# 2(a)
EC_indep = pd.Series(scenario1).quantile(0.999) - pd.Series(scenario1).mean()

# 2(b)
EC_corr = pd.Series(scenario2).quantile(0.999) - pd.Series(scenario2).mean()

# 3
# 3(a)
scenario3 = []
for _ in range(5000):
    total_loss = independent_default_loss()['default_amt'].sum()
    if total_loss < 40000000:
        scenario3.append(0)
    else:
        scenario3.append(total_loss - 40000000)
pd.Series(scenario3).hist(bins=20)
plt.show()

# 3(b)
Expected_loss = np.array(scenario3).mean()
EC_3 = pd.Series(scenario3).quantile(0.999) - pd.Series(scenario3).mean()

# 3(c)
scenario4 = []
for _ in range(5000):
    total_loss2 = single_factor_gaussian_copula_loss(copula_correlation, 5000, default_prob)['default_amt'].sum()
    if total_loss2 < 40000000:
        scenario4.append(0)
    else:
        scenario4.append(total_loss2 - 40000000)
pd.Series(scenario4).hist(bins=20)
plt.show()

Expected_loss = np.array(scenario4).mean()
EC_4 = pd.Series(scenario4).quantile(0.999) - pd.Series(scenario4).mean()

# 4
# 4(a)
scenario5 = []
for _ in range(5000):
    scenario5.append(independent_default_loss(beta=True)['default_amt'].sum())
pd.Series(scenario5).hist(bins=20)
plt.show()

# 4(b)
scenario6 = []
for _ in range(5000):
    scenario6.append(
        single_factor_gaussian_copula_loss(copula_correlation, 5000, default_prob, beta=True)['default_amt'].sum())
pd.Series(scenario6).hist(bins=20)
plt.show()
