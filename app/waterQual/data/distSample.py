
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.data import gageII, usgs, gridMET, transform
from hydroDL.post import axplot, figplot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#  figure out the overall vs sample distribution of prcp and streamflow
wqData = waterQuality.DataModelWQ('HBN')
siteNoLst = wqData.info['siteNo'].unique().tolist()

startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2019, 12, 31)

tR = pd.date_range(startDate, endDate)
dfQ = pd.DataFrame(index=tR)
dfP = pd.DataFrame(index=tR)
for siteNo in siteNoLst:
    tempQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfQ = dfQ.join(tempQ['00060_00003'].rename(siteNo))
    tempF = gridMET.readBasin(siteNo)
    dfP = dfP.join(tempF['pr'].rename(siteNo))

qA = dfQ.values
pA = dfP.values
q = wqData.q[-1, :, 0]
p = wqData.f[-1, :, 0]


fig, axes = plt.subplots(2, 2)
axes[0, 0].hist(np.log(qA.flatten()+1), density=True, bins=100)
axes[0, 0].set_ylim([0, 0.5])
axes[0, 1].hist(np.log(q+1), density=True, bins=100)
axes[0, 1].set_ylim([0, 0.5])
axes[1, 0].hist(np.log(pA.flatten()+1), density=True, bins=100)
axes[1, 0].set_ylim([0, 0.5])
axes[1, 1].hist(np.log(p+1), density=True, bins=100)
axes[1, 1].set_ylim([0, 0.5])
fig.show()


def sortData(x):
    xArrayTemp = x.flatten()
    xArray = xArrayTemp[~np.isnan(xArrayTemp)]
    xSort = np.sort(xArray)
    return xSort


qA = dfQ.values
pA = dfP.values
q = wqData.q[-1, :, 0]
p = wqData.f[-1, :, 0]

fig, ax = plt.subplots(1, 1)
x = np.log(qA+1)
xS = sortData(x)
yS = np.arange(len(xS)) / float(len(xS) - 1)
ax.plot(xS, yS, 'r')
x = np.log(q+1)
xS = sortData(x)
yS = np.arange(len(xS)) / float(len(xS) - 1)
ax.plot(xS, yS, 'b')
fig.show()


fig, ax = plt.subplots(1, 1)
x = np.log(pA+1)
xS = sortData(x)
yS = np.arange(len(xS)) / float(len(xS) - 1)
ax.plot(xS, yS, 'r')
x = np.log(p+1)
xS = sortData(x)
yS = np.arange(len(xS)) / float(len(xS) - 1)
ax.plot(xS, yS, 'b')
fig.show()
