from astropy.timeseries import LombScargle
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time


# test
xd = np.arange('1990-01-01', '2000-01-01', dtype='datetime64[D]')
xx = (xd-np.datetime64('1990-01-01')).astype(np.float)
x = xx
x = np.concatenate([x[:700], x[1400:]])
prob=0.9
x = x[np.random.rand(len(x)) >= 0.9]
y = 5 * np.cos(x*2*np.pi/365)+3 * np.cos((x+120)*2*np.pi/365*2) + \
    4 * np.cos((x+60)*2*np.pi/365/4)
# y = 5*np.sin(x*2*np.pi/365)
y = y-np.mean(y)
ls = LombScargle(x, y)
# freq = np.arange(1, len(xd))/len(xd)
freq = np.fft.fftfreq(len(xd))
power = ls.power(freq)
p = ls.model_parameters(1/365)
mat = ls.design_matrix(1/365)

ym = np.zeros([len(freq), len(x)])
yp = np.zeros([len(freq), len(x)])
for k, f in enumerate(freq[1:-1]):
    ym[k, :] = ls.model(x, f)
    yp[k, :] = ym[k, :]*ls.power(f)
yp = yp
# yy = yy-ls.offset()/2
fig, axes = plt.subplots(3, 1, figsize=(8, 6))
axes[0].plot(x, y, '--*')
axes[0].plot(x, np.sum(ym, axis=0)/2*(len(x)/len(xx)), '-r')
axes[0].set_xlabel('day')
axes[1].plot(freq, power)
axes[1].set_ylabel('power')
axes[1].set_xlabel('frequency (angular)')
axes[2].plot(1/freq, power)
axes[2].set_ylabel('power')
axes[2].set_xlabel('period (day)')
plt.tight_layout()
fig.show()


fig, axes = plt.subplots(3, 1, figsize=(10, 6))
freqAbs = np.abs(freq)
ind = np.where((freqAbs >= 0) & (freqAbs < 1/365/2))[0]
axes[0].plot(x, np.sum(ym[ind, :], axis=0), '-r', label='low freq')
# axes[1].plot(x, y, '-*b')
axes[0].legend(loc='lower right')
ind = np.where((freqAbs >= 1/365/2) & (freqAbs < 1/7))[0]
axes[1].plot(x, np.sum(yp[ind, :], axis=0), '-r', label='mid freq')
# axes[1].plot(t, y, '-*b', label='obs')
axes[1].legend(loc='lower right')
ind = np.where((freqAbs >= 1/7) & (freqAbs < 10))[0]
# axes[2].plot(x, y, '-*b', label='obs')
axes[2].plot(x, np.sum(yp[ind, :], axis=0), '-r', label='high freq')
axes[2].legend(loc='lower right')
plt.tight_layout()
fig.show()
