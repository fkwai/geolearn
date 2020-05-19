import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal


dataFile = os.path.join(kPath.dirWQ, 'tempData', 'es403723r_si_002.csv')
df = pd.read_csv(dataFile, encoding='ISO-8859-1')
t = pd.to_datetime(df['date'])
df.columns
field = 'Si_(29)  ppb'
field = 'NO3- mg/l'


# real data
dfV = df[df[field].notna().values]
nt = len(df)
x = pd.to_datetime(dfV['date']).values.astype('datetime64[D]')
y = dfV[field].values
# y = 5 * np.cos(x*2*np.pi/365)+3 * np.cos((x+75)*2*np.pi/730)
f = 2*np.pi/np.linspace(2, nt, nt)
pgram = signal.lombscargle(x, y, f)
fig, axes = plt.subplots(2, 1)
axes[0].plot(x, y, '--*')
axes[0].set_xlabel('date')
axes[0].set_ylabel('silica)')
axes[1].plot(np.log(f/2*np.pi), np.log(pgram), '-*')
axes[1].set_xlabel('log(freq)')
axes[1].set_ylabel('log(power)')
fig.show()

