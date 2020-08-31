#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import copy

# In[61]:


K0 = [3500, 3000]
mu1 = 0.01
mu2 = 0.01
sig1 = 0.25
sig2 = 0.2
rho = 0.2
r = 0.02
Dx = 0
Dy = 0
F = 100
T = 3
c = [0.0285, 0.057, 0.0855, 0.114, 0.1425, 0.171]
# c = [0.0285, 0.057]
K = [0.85, 0.85, 0.85, 0.85, 0.85, 0.85]
# K = [0.85, 0.85]
KI = 0.55
dx = K0[0] * 0.01
dy = K0[1] * 0.01
Nx = 300
Ny = 300
S1max, S1min = dx * Nx, 0
S2max, S2min = dy * Ny, 0
pp = 100  # number of time points
Nt = 6 * pp
dt = T / Nt


# In[62]:


def make_an(n, step):
    if step == 1:
        an = 1 + dt * (0.5 * r + r * n + (sig1 * n) ** 2)
    else:
        an = 1 + dt * (0.5 * r + r * n + (sig2 * n) ** 2)
    return an


def make_bn(n, step):
    if step == 1:
        bn = -((sig1 * n) ** 2) * dt * 0.5
    else:
        bn = -((sig2 * n) ** 2) * dt * 0.5
    return bn


def make_cn(n, step):
    if step == 1:
        cn = -((sig1 * n) ** 2) * dt * 0.5 - r * n * dt
    else:
        cn = -((sig2 * n) ** 2) * dt * 0.5 - r * n * dt
    return cn


def make_dn(m, n, l):
    dm = m[l, n] + (rho * sig1 * sig2 * n * l * dt * 0.5) * (
                m[l + 1, n + 1] + m[l - 1, n - 1] - m[l + 1, n - 1] - m[l - 1, n + 1]) * 0.25
    return dm


make_an = np.vectorize(make_an)
make_bn = np.vectorize(make_bn)
make_cn = np.vectorize(make_cn)


# In[63]:


def make_v0_2(x, y, KI_type):
    x_r = x / K0[0]
    y_r = y / K0[1]
    if KI_type == 0:
        if (x_r <= KI or y_r <= KI):
            return min([x_r, y_r]) * F
        else:
            return (1 + c[-1]) * F
    else:
        if (x_r <= K[-5] or y_r <= K[-5]):
            return min([x_r, y_r]) * F
        else:
            return (1 + c[-1]) * F


make_v0_2 = np.vectorize(make_v0_2)


def BC_condition(m):
    # u(t, 0, y)
    x0 = 2 * m[:, 1] - m[:, 2]
    # u(t, xmax, y)
    x_max = 2 * m[:, -2] - m[:, -3]
    # u(t, x, 0)
    y0 = 2 * m[1, :] - m[2, :]
    # u(t, x, ymax)
    y_max = 2 * m[-2, :] - m[-3, :]
    m[:, 0] = x0
    m[:, -1] = x_max
    m[0, :] = y0
    m[-1, :] = y_max
    # print(x0, x_max, y0, y_max)
    p1 = 0
    p2 = 0
    p3 = (x_max[-2] + y_max[-2]) / 2
    p4 = 0
    m[0, 0] = p1
    m[0, -1] = p2
    m[-1, -1] = p3
    m[-1, 0] = p4
    return m


def make_v0(S1min, S1max, Nx, S2min, S2max, Ny, KI_type):
    # KI_type = 1, non-KI_type = 0
    Sx = np.linspace(S1min, S1max, Nx + 1)
    Sy = np.linspace(S2min, S2max, Ny + 1)
    Sx, Sy = np.meshgrid(Sx, Sy)
    v0 = make_v0_2(Sx, Sy, KI_type)
    return v0


# In[64]:


def step_1_span(m, Nx, Ny):
    n = np.arange(Nx + 1)[1:-1]
    a = make_an(n, 1)
    b = make_bn(n, 1)
    c = make_cn(n, 1)
    a[0] = 2 * b[0] + a[0]
    c[0] = c[0] - b[0]
    b[-1] = b[-1] - c[-1]
    a[-1] = a[-1] + 2 * c[-1]
    b[0] = 0
    c[-1] = 0
    l = np.arange(Ny + 1)[1:-1]
    n, l = np.meshgrid(n, l)
    d = make_dn(m, n, l)
    # d = d.T
    v1 = np.zeros((m.shape[1], m.shape[0]))
    n = np.arange(a.shape[0])
    x = np.zeros((d.shape[0], d.shape[0]))

    def Thomas_Al(i):
        nonlocal d
        if i == 0:
            pass
        else:
            a[i] = a[i] - (b[i] / a[i - 1]) * c[i - 1]
            d[i, :] = d[i, :] - (b[i] / a[i - 1]) * d[i - 1, :]
        return d

    Thomas_Al = np.vectorize(Thomas_Al)

    def Thomas_Al_2(i):
        nonlocal x
        j = a.shape[0] - (i + 1)
        if j == (a.shape[0] - 1):
            x[j, :] = d[j, :] / a[j]
        else:
            x[j, :] = (d[j, :] - c[j] * x[j + 1, :]) / a[j]
        return x

    Thomas_Al_2 = np.vectorize(Thomas_Al_2)
    try:
        Thomas_Al(n)  # 에러가 발생할 가능성이 있는 코드
    except ValueError:
        pass
    try:
        Thomas_Al_2(n)  # 에러가 발생할 가능성이 있는 코드
    except ValueError:
        pass
    v1[1:-1, 1:-1] = x
    return v1


# In[65]:


def step_2_span(m, Nx, Ny):
    n = np.arange(Nx + 1)[1:-1]
    a = make_an(n, 2)
    b = make_bn(n, 2)
    c = make_cn(n, 2)
    a[0] = 2 * b[0] + a[0]
    c[0] = c[0] - b[0]
    b[-1] = b[-1] - c[-1]
    a[-1] = a[-1] + 2 * c[-1]
    b[0] = 0
    c[-1] = 0
    n = np.arange(Nx + 1)[1:-1]
    l = np.arange(Ny + 1)[1:-1]
    l, n = np.meshgrid(n, l)
    d = make_dn(m, n, l)
    # d = d.T
    v1 = np.zeros((m.shape[1], m.shape[0]))
    n = np.arange(a.shape[0])
    x = np.zeros((d.shape[0], d.shape[0]))

    def Thomas_Al(i):
        nonlocal d
        if i == 0:
            pass
        else:
            a[i] = a[i] - (b[i] / a[i - 1]) * c[i - 1]
            d[i, :] = d[i, :] - (b[i] / a[i - 1]) * d[i - 1, :]
        return d

    Thomas_Al = np.vectorize(Thomas_Al)

    def Thomas_Al_2(i):
        nonlocal x
        j = a.shape[0] - (i + 1)
        if j == (a.shape[0] - 1):
            x[j, :] = d[j, :] / a[j]
        else:
            x[j, :] = (d[j, :] - c[j] * x[j + 1, :]) / a[j]
        return x

    Thomas_Al_2 = np.vectorize(Thomas_Al_2)
    try:
        Thomas_Al(n)  # 에러가 발생할 가능성이 있는 코드
    except ValueError:
        pass
    try:
        Thomas_Al_2(n)  # 에러가 발생할 가능성이 있는 코드
    except ValueError:
        pass
    v1[1:-1, 1:-1] = x
    return v1


# In[66]:


def update_plane1(m1, m2, Nx, Ny):
    Sx = list(np.linspace(S1min, S1max, Nx + 1) / K0[0])
    Sy = list(np.linspace(S2min, S2max, Ny + 1) / K0[1])
    x_index = Sx.index(KI)
    y_index = Sy.index(KI)
    # print(KI, x_index, y_index)
    # make m*
    m1_star = step_1_span(m1, Nx, Ny)
    m1_star = BC_condition(m1_star)

    m2_star = step_1_span(m2, Nx, Ny)
    update_plane = copy.deepcopy(m1_star)
    small_plane = m2_star[y_index:, x_index:]
    update_plane[y_index:, x_index:] = small_plane
    # m2_star = BC_condition(m2_star)
    # m2_star[:56, :56] = m1_star[:56, :56]
    update_plane = BC_condition(update_plane)
    return m1_star, update_plane


# In[67]:


def update_plane2(m1, m2, Nx, Ny):
    Sx = list(np.linspace(S1min, S1max, Nx + 1) / K0[0])
    Sy = list(np.linspace(S2min, S2max, Ny + 1) / K0[1])
    x_index = Sx.index(KI)
    y_index = Sy.index(KI)

    # make m*
    m1_star = step_2_span(m1, Nx, Ny)
    m1_star = BC_condition(m1_star)

    m2_star = step_2_span(m2, Nx, Ny)
    update_plane = copy.deepcopy(m1_star)
    small_plane = m2_star[y_index:, x_index:]
    update_plane[y_index:, x_index:] = small_plane
    # m2_star = BC_condition(m2_star)
    # m2_star[:56, :56] = m1_star[:56, :56]
    update_plane = BC_condition(update_plane)
    return m1_star, update_plane


# In[68]:


def from_m_to_m1(m1, m2, Nx, Ny):
    v1, v2 = update_plane1(m1, m2, Nx, Ny)
    v3, v4 = update_plane2(v1, v2, Nx, Ny)
    return v3, v4


# In[69]:


def barrier(m, B, C):
    # B = Barrier
    # C = Coupon
    Sx = list(np.linspace(S1min, S1max, Nx + 1) / K0[0])
    Sy = list(np.linspace(S2min, S2max, Ny + 1) / K0[1])
    x_index = Sx.index(B)
    y_index = Sy.index(B)
    m[y_index:, x_index:] = (1 + C) * F
    return m


# In[70]:


def OS_FDM(Nt, Nx, Ny):
    price1 = np.zeros((Nt + 1, Nx + 1, Ny + 1))
    price0 = np.zeros((Nt + 1, Nx + 1, Ny + 1))

    price1[0, :, :] = make_v0(S1min, S1max, Nx, S2min, S2max, Ny, 1)  # KI plane
    price0[0, :, :] = make_v0(S1min, S1max, Nx, S2min, S2max, Ny, 0)  # n-KI plane

    early_redemption = np.array([100, 200, 300, 400, 500])
    n = np.arange(Nt)

    # temp_plane = make_v0(S1min, S1max, Nx, S2min, S2max, Ny, 1) #KI plane

    def cal(i):
        nonlocal price1
        nonlocal price0
        nonlocal early_redemption
        # nonlocal temp_plane
        i += 1
        print(i)
        mat1, mat2 = from_m_to_m1(price1[i - 1, :, :], price0[i - 1, :, :], Nx, Ny)
        price1[i, :, :] = mat1
        price0[i, :, :] = mat2

        if i in early_redemption:
            j = int(i / 100)
            price1[i, :, :] = barrier(mat1, K[5 - j], c[5 - j])
            price0[i, :, :] = barrier(mat2, K[5 - j], c[5 - j])
        return price1, price0

    cal = np.vectorize(cal)

    try:
        cal(n)
    except ValueError:
        pass
    return price1, price0


# In[56]:


import timeit

start = timeit.default_timer()

els1, els2 = OS_FDM(Nt, Nx, Ny)

stop = timeit.default_timer()
print(stop - start)

# In[57]:


els1[-1, :, :][100, 100]

# In[58]:


els2[-1, :, :][100, 100]

# In[59]:


from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

Sx = np.linspace(S1min, S1max, Nx + 1)
Sy = np.linspace(S2min, S2max, Ny + 1)
Sx, Sy = np.meshgrid(Sx, Sy)
ax.plot_surface(Sx, Sy, els2[-1, :, :], rstride=1, cstride=1,
                cmap='winter', edgecolor='none')

plt.show()

# In[ ]:
