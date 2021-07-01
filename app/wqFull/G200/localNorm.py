import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
dictSiteName = 'dictWeathering.json'
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['k12']

dataName = 'weathering'
DF = dbBasin.DataFrameBasin(dataName)
trainSet = 'rmYr5'
testSet = 'pkYr5'

# input
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
subset = trainSet
DF = dbBasin.localNorm(DF, subset=trainSet)

DM = dbBasin.DataModelBasin(DF)

# plot
code = '00915'
nfy, nfx = (4, 3)
bins = 20
data0 = DF.c
data1 = DF.extractSubset(DF.c, subsetName=trainSet)
data2 = DF.extractSubset(DF.c, subsetName=testSet)
dataLst = [data0, data1, data2]
titleLst = ['all', 'train', 'test']
for data, title in zip(dataLst, titleLst):
    fig, axes = plt.subplots(nfy, nfx)
    for k, siteNo in enumerate(DF.siteNoLst):
        j, i = utils.index2d(k, nfy, nfx)
        ax = axes[j, i]
        ic = DF.varC.index(code+'-N')
        _ = ax.hist(data[:, k, ic], bins=bins, density=True)
        shortName = usgs.codePdf.loc[code]['shortName']
        nData = np.sum(~np.isnan(data[:, k]))
        titleStr = '{} {}'.format(siteNo, nData)
        axplot.titleInner(ax, titleStr, top=False)
    fig.suptitle(title)
    fig.show()

# plot overall
icLst1 = [DF.varC.index(code) for code in codeSel]
icLst2 = [DF.varC.index(code+'-N') for code in codeSel]
fig, axes = plt.subplots(2, 3)
for k, data in enumerate(dataLst):
    axes[0, k].hist(data[:, :, icLst1].flatten(), bins=bins, density=True)
    axes[1, k].hist(data[:, :, icLst2].flatten(), bins=bins, density=True)
    fig.show()

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, DF.t, data2[:, 0, icLst2[0]])
fig.show()
