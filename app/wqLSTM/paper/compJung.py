

import matplotlib.dates as mdates
import random
import scipy
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import matplotlib.gridspec as gridspec

# data comp
# siteComp = ['04208000', '04212100', '03271601',
#             '04193500', '03150000', '04176500', '04199500']
# nameComp = ['CY', 'GD', 'GM', 'MM', 'MS', 'RS', 'VM']

# # GM not in USGS
# siteComp = ['04208000', '04212100',
#             '04193500', '03150000', '04176500', '04199500']
# nameComp = ['CY', 'GD',  'MM', 'MS', 'RS', 'VM']
# DFJ = dbBasin.DataFrameBasin.new('Jung2020', siteComp, varC=['00618'])

# siteComp = ['04208000', '04212100',
#             '04193500', '03150000', '04176500', '04199500']
# nameComp = ['CY', 'GD',  'MM', 'MS', 'RS', 'VM']
# # plot data
# dataLst = list()
# labLst = list()
# for s in siteComp:
#     k = DFJ.siteNoLst.index(s)
#     k
#     dataLst.append(DFJ.c[:, k, 0])
#     labLst.append(nameComp[k])
# dataPlot = np.stack(dataLst, axis=-1)

# fig, axes = figplot.multiTS(DFJ.t, dataPlot)
# for k, ax in enumerate(axes):
#     axplot.titleInner(ax, nameComp[k], offset=0.2)
# fig.show()

siteComp = ['04208000', '04212100',
            '04193500', '03150000', '04176500', '04199500']
nameComp = ['CY', 'GD',  'MM', 'MS', 'RS', 'VM']
DF = dbBasin.DataFrameBasin('G200')
indS = list()
siteLst = list()
nameLst = list()
for s, n in zip(siteComp, nameComp):
    if s in DF.siteNoLst:
        indS.append(DF.siteNoLst.index(s))
        siteLst.append(s)
        nameLst.append(n)
    else:
        print(s)


# load TS
ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=1000)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


code = '00618'
codeStr = usgs.codePdf.loc[code]['shortName']
outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
saveFolder = os.path.join(outFolder, code)
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
# ts map
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

lat, lon = DF.getGeo()
indC = codeLst.index(code)
importlib.reload(figplot)
importlib.reload(axplot)
importlib.reload(mapplot)

yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)
t = DF.t
yr = DF.t.astype('M8[Y]').astype(str).astype(int)
indT = np.where(np.in1d(yr, np.array(yrLst)))[0]

# plot TS
for siteNo, name in zip(siteLst, nameLst):
    ind = DF.siteNoLst.index(siteNo)
    dataPlot = [yW[:, ind, indC], yP[:, ind, indC],
                DF.c[:, ind, DF.varC.index(code)]]
    a = yW[indT, ind, indC]
    b = yP[indT, ind, indC]
    o = DF.c[indT, ind, indC]
    rRMSE1 = np.sqrt(np.nanmean(((a-o)/o)**2))
    rRMSE2 = np.sqrt(np.nanmean(((b-o)/o)**2))
    MPE1 = np.nanmean((o-a)/o)*100
    MPE2 = np.nanmean((o-b)/o)*100

    cLst = 'kbr'
    legLst = ['WRTDS {:.2f} {:.2f}'.format(rRMSE1, MPE1),
              'LSTM {:.2f} {:.2f}'.format(rRMSE2, MPE2),
              'Obs']
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, ny, wspace=0)
    axP0 = figP.add_subplot(gsP[0, 0])
    axPLst = [axP0]
    for k in range(1, ny):
        axP = figP.add_subplot(gsP[0, k], sharey=axP0)
        axPLst.append(axP)
    axP = np.array(axPLst)
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst, legLst=legLst)
    saveFolder = r'C:\Users\geofk\work\waterQuality\paper\G200\Jung2020'
    titleStr = '{} {}'.format(siteNo, name)
    figP.suptitle('{} of site {}'.format(codeStr, siteNo))
    figP.tight_layout()
    figP.show()
    figP.savefig(os.path.join(saveFolder, '{}_{}'.format(siteNo, trainSet)))
