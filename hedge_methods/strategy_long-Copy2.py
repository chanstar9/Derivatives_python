import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
from scipy.stats import norm
import copy
import time

m = 20
df = pd.read_excel('POSCO4.xlsx', sheet_name='Sheet2')
df.set_index('date', inplace=True)
posco_return = np.log(df.POSCO) - np.log(df.POSCO.shift(1))
posco_return[0] = 0
hist_vol = posco_return.rolling(m + 1).std() * np.sqrt(252)
skew = (((df - df.mean()) / (df.std())) ** 3).mean()

pred_vol = []
n = 1
for i in range(posco_return.shape[0] - m):
    model = arch_model(posco_return[i:m + i] - posco_return[i:m + i].mean(), mean='Zero', vol='GARCH', p=1, q=1)
    model_fit = model.fit()
    yhat = model_fit.forecast(horizon=n)

    pred_vol.append(yhat.variance.values[-1, :n].mean())
pred_vol = np.sqrt(np.array(pred_vol) * 252)

# Parameters for simulation
S0 = df["2016-7"].iloc[0][0]  # float(input('Enter initial stock price (Default = 49) : '))
R = 0.02  # float(input('Enter risk-free rate (Default = 0.05) : '))
K = df["2016-7"].iloc[0][0] * 1.1  # float(input('Enter strike price (Default = 50) : '))
TTM = float(125 / 252)  # float(eval(input('Enter initial Time To Maturity (Default = 20/52) : ')))
frq = 126  # int(eval(input('Enter hedging frequency (integer) :')))+1 # 매 시간마다 path를 생성
CONTS = 10000  # number of contracts
tc = 0.001  # transaction cost
NUMSIM = 1
start_vol = posco_return[:'2016-6'].std() * np.sqrt(252)


def vol(pred_vol, hist_vol, df):  # df['2016-7':]
    sign = np.array(df['2016-7':])[1:] - np.array(df['2016-7':])[:-1]
    pvcopy = np.array(pred_vol[-sign.shape[0]:]).reshape(-1, 1)
    hvcopy = np.array(hist_vol[-sign.shape[0] - 1:-1]).reshape(-1, 1)
    volvol = []
    for i in range(sign.shape[0]):
        if sign[i] <= 0:
            if skew[0] <= 0:
                volvol.append(max(pvcopy[i][0], start_vol))
            else:
                volvol.append(min(pvcopy[i][0], start_vol))
        else:
            if skew[0] <= 0:
                volvol.append(min(pvcopy[i][0], start_vol))
            else:
                volvol.append(max(pvcopy[i][0], start_vol))

    return volvol


volvol = vol(pred_vol, hist_vol, df)
volvol = np.array(volvol)
volvol = np.r_[start_vol, volvol]


def ND1(s, t, start_vol):
    if t != 0:
        d1 = (1 / (start_vol * np.sqrt(t))) * (np.log(s / K) + (R + start_vol ** 2 / 2) * t)
        return norm.cdf(d1)
    elif s >= K:
        return 1
    else:
        return 0


def ND2(s, t, start_vol):
    d2 = (1 / (start_vol * np.sqrt(t))) * (np.log(s / K) + (R + start_vol ** 2 / 2) * t) - start_vol * np.sqrt(t)
    return norm.cdf(d2)


def striker(s):
    if s >= K:
        return 1
    else:
        return 0


def timehedger(i):
    for j in range(1, frq - 1):
        tradevolume = nd1[j] - nd1[j - 1]
        cumcost[i, j] = cumcost[i, j - 1] * np.exp(R * dt)
        cumcost[i, j] = cumcost[i, j] + tradevolume * CONTS * stockpath[i, j]
        cumcost[i, j] = cumcost[i, j] + max(-tradevolume, 0) * CONTS * stockpath[i, j] * tc


TTMvec = np.linspace(TTM, 0, frq - 1).reshape(1, -1)  # (1,4368) matrix 이다.
dt = TTM / (frq - 1)
stockpath = np.array(df[-(frq - 1):]).reshape(1, -1)

d1 = (1 / (volvol[:-1] * np.sqrt(TTMvec[0, :-1]))) * (
            np.log(stockpath[0][:-1] / K) + (R + 0.5 * volvol[:-1] ** 2) * TTMvec[0, :-1])

nd1 = norm.cdf(d1)
ender = striker(stockpath[0, -1])
nd1 = np.r_[nd1, ender]
cumcost = np.zeros((NUMSIM, frq - 1))
cumcost[:, 0] = nd1[0] * CONTS * (stockpath[0, 0] * (1 + tc))

s_nd1 = copy.deepcopy(nd1)

for i in range(NUMSIM):
    timehedger(i)
s_cum = copy.deepcopy(cumcost)

finalcost = []
for i in range(NUMSIM):
    if nd1[-1] == 1:
        finalcost.append(cumcost[i, -1] - K * CONTS)
    else:
        finalcost.append(cumcost[i, -1])

strategy_cost = float(finalcost[0])
c = S0 * ND1(S0, TTM, start_vol) - K * np.exp(-R * TTM) * ND2(S0, TTM, start_vol)
print("BS model price is")
print(c * CONTS)
print("strategy performace : ", finalcost / (c * CONTS))

d1 = (1 / (start_vol * np.sqrt(TTMvec[0, :-1]))) * (
            np.log(stockpath[0][:-1] / K) + (R + 0.5 * start_vol ** 2) * TTMvec[0, :-1])
nd1 = norm.cdf(d1)
ender = striker(stockpath[0, -1])
nd1 = np.r_[nd1, ender]

cumcost = np.zeros((NUMSIM, frq - 1))
cumcost[:, 0] = nd1[0] * CONTS * (stockpath[0, 0] * (1 + tc))

t_nd1 = copy.deepcopy(nd1)

for i in range(NUMSIM):
    timehedger(i)

t_cum = copy.deepcopy(cumcost)
finalcost = []
for i in range(NUMSIM):
    if nd1[-1] == 1:
        finalcost.append(cumcost[i, -1] - K * CONTS)
    else:
        finalcost.append(cumcost[i, -1])

print("hist performace : ", finalcost / (c * CONTS))

c = S0 * ND1(S0, TTM, start_vol * 1.15) - K * np.exp(-R * TTM) * ND2(S0, TTM, start_vol * 1.15)
SS0 = K
SS_end = df.iloc[-1][0]
print("BS model price with margin is")
print(c * CONTS)
histvol_cost = float(finalcost[0])
if SS0 > SS_end:
    print("OTM")
else:
    print("ITM")
print("strategy cost : ", strategy_cost)
print("traditional hedge cost : ", histvol_cost)

plt.plot(s_nd1, label="strategy delta")
plt.plot(t_nd1, label='BS delta')
plt.legend()
# plt.plot(stockpath)


plt.plot(s_cum.T, label="strategy cum cost")
plt.plot(t_cum.T, label="BS cum cost")
plt.legend()

plt.plot(df["2016-7":])

sign = np.array(df['2016-7':])[1:] - np.array(df['2016-7':])[:-1]
pvcopy = np.array(pred_vol[-sign.shape[0]:]).reshape(-1, 1)
hvcopy = np.array(hist_vol[-sign.shape[0] - 1:-1]).reshape(-1, 1)
plt.plot(pvcopy, color='r', label="GARCH(1,1)")
plt.plot(hvcopy, label="moving vol")
plt.plot(np.ones((124, 1)) * start_vol, label="historical constant vol")
plt.legend()

plt.plot(volvol)

sign = np.array(df['2016-7':])[1:] - np.array(df['2016-7':])[:-1]
pvcopy = np.array(pred_vol[-sign.shape[0]:]).reshape(-1, 1)
hvcopy = np.array(hist_vol[-sign.shape[0] - 1:-1]).reshape(-1, 1)
plt.plot(pvcopy, color='r', label="GARCH(1,1)")
plt.plot(hvcopy, label="moving vol")
plt.plot(np.ones((124, 1)) * start_vol, label="historical constant vol")
plt.plot(volvol, label="mix_vol")
plt.legend()

plt.plot(df["2016-7":])
