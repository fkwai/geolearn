import scipy
import time
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL.master import basins
from hydroDL.data import gageII, usgs, gridMET
from hydroDL import kPath, utils
import os
import pandas as pd
import numpy as np
from hydroDL import kPath

fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.newC)
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE', 'CLASS'], siteNoLst=siteNoLstAll)
dfCrd = gageII.updateCode(dfCrd)
sd = np.datetime64('1979-01-01')

# load all data
dictC = dict()
dictCF = dict()
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfC, dfCF = usgs.readSample(siteNo, codeLst=codeLst, startDate=sd, flag=2)
    dictC[siteNo] = dfC
    dictCF[siteNo] = dfCF
dictQ = dict()
for k, siteNo in enumerate(siteNoLstAll):
    print(k, siteNo)
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    dictQ[siteNo] = dfQ

# app\waterQual\stableSites\countSiteYear.py

# calculate interval
intMatC = np.full([len(siteNoLstAll), len(codeLst), 4], np.nan)
for k, siteNo in enumerate(siteNoLstAll):
    dfC = dictC[siteNo]
    print('\t {}/{}'.format(k, len(siteNoLstAll)), end='\r')
    for j, code in enumerate(codeLst):
        tt = dfC[code].dropna().index.values
        if len(tt) > 1:
            dt = tt[1:]-tt[:-1]
            dd = dt.astype('timedelta64[D]').astype(int)
            intMatC[k, j, 0] = len(tt)
            intMatC[k, j, 1] = np.percentile(dd, 25)
            intMatC[k, j, 2] = np.percentile(dd, 50)
            intMatC[k, j, 3] = np.percentile(dd, 75)
fig, ax = plt.subplots(1, 1)
for code in codeLst:
    ic = codeLst.index(code)
    v = intMatC[:, ic, 2]
    vv = np.sort(v[~np.isnan(v)])
    x = np.arange(len(vv))
    ax.plot(x, vv, label=code)
ax.set_ylim([0, 100])
ax.set_xlim([0, 1000])
ax.legend()
fig.show()

[indS, indC] = np.where((intMatC[:, :, 0] > 150) & (intMatC[:, :, 2] < 40))
len(np.unique(indS))

# use convolve to count # samples within one year
siteNo = siteNoLstAll[0]
code = '00915'
df = dictC[siteNo][code].dropna()

sd = np.datetime64('1979-01-01')
ed = np.datetime64('2019-12-31')
td = pd.date_range(sd, ed)
dfD = pd.DataFrame({'date': td}).set_index('date')
dfD = dfD.join(df)
