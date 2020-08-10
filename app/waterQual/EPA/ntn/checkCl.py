import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt


# varC = usgs.varC
varC = ['00940']
siteNoLst = ['0422026250', '04232050', '0423205010']
tabG = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)

# read data
siteNo = siteNoLst[2]
dfC = usgs.readSample(siteNo, codeLst=varC)
dfQ = usgs.readStreamflow(siteNo)
dfF = gridMET.readBasin(siteNo)
ntnFolder = os.path.join(kPath.dirData, 'EPA', 'NTN', 'usgs', 'weeklyRaw')
dfP = pd.read_csv(os.path.join(ntnFolder, siteNo), index_col='date')

# convert to weekly
t = pd.date_range(start='1979-01-01', end='2019-12-30', freq='W-TUE')
td = pd.date_range(t[0], t[-1])    
df = pd.DataFrame({'date': td}).set_index('date')
df = df.join(dfC)
df = df.join(dfQ)
df = df.join(dfF)
df = df.rename(columns={'00060_00003': '00060'})
dfW = df.resample('W-TUE').mean()
dfW = dfW.join(dfP)

area = tabG.loc[siteNo]['DRAIN_SQKM']
load1 = dfW['pr']*area*dfW['Cl']
load2=dfW['00060']*dfW['00940']

fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
ax.plot(load1,'r--*')
ax2.plot(load2,'b--*')
fig.show()


# plot
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()
ax.plot(dfP['Cl'], 'b--*')
ax.plot(dfP['Cl'], 'g--*')
ax2.plot(dfCW[varC], 'r*')
ax2.plot(dfC[varC], 'm*')
fig.show()
