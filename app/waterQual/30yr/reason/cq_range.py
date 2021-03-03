from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from hydroDL.utils.stat import calPercent
from hydroDL.app.waterQuality.wqRela import analRange

codeLst = sorted(usgs.newC)


code = '00945'
siteNo = '10172200'
df = waterQuality.readSiteTS(siteNo, varLst=['00060']+codeLst)

t = df.index.values
q = df['00060'].values
ql = np.log(q+1)
c = df[code].values

fig, axes = plt.subplots(2, 1)
axplot.plotTS(axes[0], t, q, cLst='b', styLst='-')
axplot.plotTS(axes[1], t, c, cLst='r')
fig.show()

# percentile
x = ql
y = c
bIn = False
bOut = True
rIn = np.linspace(0, 1, 11)
rOut = [0.05, 0.95]

xIn, yIn, xOut, yOut = analRange(
    ql, c, rIn, rOut, bIn=bIn, bOut=bOut)

# plot
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
xp = (xIn[1:]+xIn[:-1])/2
xe = (xIn[1:]-xIn[:-1])/2
axes[0].plot(ql, c, '*r')
# axes[0].plot(xp, yOut, '-*b')
for i in range(len(rOut)):
    axes[0].errorbar(xp, yOut[:, i], xerr=xe, fmt='-b', capsize=2)
yp = (yIn[1:]+yIn[:-1])/2
ye = (yIn[1:]-yIn[:-1])/2
axes[1].plot(ql, c, '*r')
# axes[1].plot(xOut, yp, '-*g')
for i in range(len(rOut)):
    axes[1].errorbar(xOut[:, i], yp, yerr=ye, fmt='-g', capsize=2)
fig.show()
