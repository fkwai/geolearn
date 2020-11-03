from hydroDL.post import axplot, figplot
from hydroDL.new import fun
from hydroDL.app import waterQuality
import importlib
import matplotlib.pyplot as plt
from scipy.stats import gamma
import numpy as np
import random

importlib.reload(fun)
kLst = [1, 5, 10]
rLst = [0.5, 0.2, 0.1]
# kLst = [1, 2, 5, 10, 20]
# flow duration curves
t = np.arange(365)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for k in kLst:
    q = fun.fdc(t, k)
    print(np.sum(q))
    ax.plot(t, q, label='a={}'.format(k))
ax.legend()
ax.set_title('flow duration curve')
fig.show()

# kate's model
# rLst = [0, 0.1, 0.2, 0.5, 1, 2]
t = np.arange(365)
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for r in rLst:
    ct = fun.kate(t, r)
    ax.plot(t, ct, label='tw={}'.format(r))
ax.legend()
ax.set_title('concentration with travel time')
fig.show()

# prcp
t = np.arange('2000-01-01', '2005-01-01', dtype='datetime64[D]')
x = (t-np.datetime64('1990-01-01')).astype(np.float)
p = 10 * np.cos(x*2*np.pi/365) +\
    10 * np.cos((x+120)*np.pi/365*4)
p[p < 0] = 0
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.plot(t, p)
fig.show()


# # prcp - real world
siteNo = '401733105392404'
code = '00955'
dfO = waterQuality.readSiteTS(siteNo, ['runoff', 'pr', code])
t = dfO.index.values
p = dfO['pr'].values
q = dfO['runoff'].values
fig, axes = plt.subplots(3, 1, figsize=(12, 6))
axes[0].plot(t, p)
axes[1].plot(t, dfO['runoff'].values)
axes[2].plot(t, dfO[code].values, '*')
fig.show()
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(dfO['runoff'].values, dfO[code].values, '*')
fig.show()

# calculate concentration curve
nf = len(kLst)
rho = 365
qLst = list()
cLst = list()
for k in range(nf):
    tt = np.arange(rho).T
    qc = fun.fdc(tt, kLst[k])
    cc = qc*fun.kate(tt, rLst[k])
    q = np.convolve(qc, p, 'valid')
    c = np.convolve(cc, p, 'valid')/q
    qLst.append(q)
    cLst.append(c)

# plot
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for k in range(nf):
    ax.plot(t[rho-1:], qLst[k])
fig.show()
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
for k in range(nf):
    ax.plot(t[rho-1:], cLst[k])
fig.show()
fig, axes = plt.subplots(nf, 1, figsize=(12, 6))
for k in range(nf):
    axes[k].plot(qLst[k], cLst[k], '*')
fig.show()

# out
qMat = np.stack(qLst, axis=-1)
cMat = np.stack(cLst, axis=-1)
fig, axes = plt.subplots(2, 1, figsize=(12, 6))
axes[0].plot(t[rho-1:], np.mean(qMat, axis=1))
axes[1].plot(t[rho-1:], np.mean(cMat, axis=1))
fig.show()

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(np.mean(qMat, axis=1), np.mean(cMat, axis=1), '*')
fig.show()
