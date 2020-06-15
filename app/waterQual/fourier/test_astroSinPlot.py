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
# x = np.concatenate([x[:700], x[1400:]])
pLst = [0, 0.01, 0.1, 0.5]
# pLst = [0, 0.7, 0.8, 0.9]
fig, axes = plt.subplots(len(pLst), 1, figsize=(8, 6))
for kk, p in enumerate(pLst):
    x = x[np.random.rand(len(x)) >= p]
    y = 5 * np.cos(x*2*np.pi/365)+4 * np.cos((x+120)*2*np.pi/365*2) + \
        4 * np.cos((x+60)*2*np.pi/365/4)
    # y = 5*np.sin(x*2*np.pi/365)
    y = y-np.mean(y)
    ls = LombScargle(x, y)
    # freq = np.arange(1, len(xd))/len(xd)
    freq = np.fft.fftfreq(len(xd))[1:]
    power = ls.power(freq)
    ym = np.zeros([len(freq), len(xx)])
    yp = np.zeros([len(freq), len(xx)])
    for k, f in enumerate(freq):
        ym[k, :] = ls.model(xx, f)
        # yp[k, :] = ym[k, :]*np.sqrt(ls.power(f))
        yp[k, :] = ym[k, :]*ls.power(f)
    axes[kk].plot(x, y, '--*')
    axes[kk].plot(xx, np.sum(ym/2*(1-p), axis=0), '-r')
    axes[kk].set_title('{}% missing data'.format(p*100))
plt.tight_layout()
fig.show()
