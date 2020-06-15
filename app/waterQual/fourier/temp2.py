

from scipy import fft, ifft
import scipy.signal as signal
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt

import importlib

import pandas as pd
import numpy as np
import os
import time

import scipy.signal as signal


# test
xd = np.arange('1990-01-01', '2000-01-01', dtype='datetime64[D]')
x = (xd-np.datetime64('1990-01-01')).astype(np.float)
# x = np.concatenate([x[:700], x[1400:]])
# x = x[np.random.rand(len(x)) >= 0.95]
# y = 5 * np.cos(x*2*np.pi/365)+3 * np.cos((x+120)*2*np.pi/365*2) + \
#     4 * np.cos((x+75)*2*np.pi/365/4)
y = 5 * np.cos(x*2*np.pi/365)
f = 2*np.pi/np.linspace(1, len(xd), len(xd))
pgram = signal.lombscargle(x, y, f)



fig, axes = plt.subplots(3, 1, figsize=(8, 6))
axes[0].plot(x, y, '--*')
axes[0].set_xlabel('day')
# axes[1].plot(1/f, pgram)
axes[1].plot(f, pgram)
axes[1].set_ylabel('power')
axes[1].set_xlabel('frequency (angular)')
axes[2].plot(1/f*2*np.pi, pgram)
axes[2].set_ylabel('power')
axes[2].set_xlabel('period (day)')
plt.tight_layout()
fig.show()
