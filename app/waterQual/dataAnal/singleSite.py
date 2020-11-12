from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

import pandas as pd
import numpy as np
import os
import time
import json

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')

# countMat = np.load(os.path.join(dirInv, 'matCountDaily.npy'))
countMat = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

# get siteNo
count = np.sum(countMat, axis=1)
dfCount = pd.DataFrame(index=siteNoLstAll, columns=codeLst, data=count)
code = '00915'
dfCount.sort_values(by=[code])

siteNo = '09392000'
varLst = codeLst+gridMET.varLst+['00060']
df = waterQuality.readSiteTS(siteNo, varLst=varLst)

# hist
fig, axes = plt.subplots(2, 2)
axes[0, 0].hist(df['00955'].values, density=True, bins=50)
axes[0, 1].hist(df['00095'].values, density=True, bins=50)
axes[1, 0].hist(df['00060'].values, density=True, bins=50)
axes[1, 1].hist(df['etr'].values, density=True, bins=50)
fig.show()

# lg
code = '00915'
tt = df.index.values
xx = (tt.astype('datetime64[D]') -
      np.datetime64('1979-01-01')).astype(np.float)
dfD = df[df[code].notna().values]
# dfD['log-'+code]=np.log(dfD[code]+1)
t = dfD.index.values
x = (t.astype('datetime64[D]') -
     np.datetime64('1979-01-01')).astype(np.float)
y = dfD[code].values
# y = dfD['log-'+code].values
nt = len(tt)
freq = np.fft.fftfreq(nt)[1:]
ind = np.where((1/freq >= 0) & (1/freq < 1000))[0]
freq = freq[ind]
ls = LombScargle(x, y)
power = ls.power(freq)
p = ls.false_alarm_probability(power)
fig, ax = plt.subplots(1, 1)
# ax.plot(1/freq, 1-p, '-r', label='baluev probability')
ax.plot(1/freq, power, '-*b', label='power')
fig.show()

waterQuality.calPower(code,df)
# # ts
# fig, ax = plt.subplots(1, 1)
# ax.plot(df.index, df['00955'].values, '*')
# fig.show()
