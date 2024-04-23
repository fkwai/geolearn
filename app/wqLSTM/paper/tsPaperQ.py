
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


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


# LSTM corr
ep = 500
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

# WRTDS corr
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW2 = np.load(corrFile2)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

# load TS
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=500)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']


dictPlot = dict()

# dictPlot['00915'] = ['12323800', '08057200', '02175000']
# dictPlot['00915'] = ['12323800', '07241550', '07241000']
dictPlot['00915'] = ['12323600','09251000','07227500']
code = '00915'
siteLst = dictPlot[code]
codeStr = usgs.codePdf.loc[code]['shortName']
outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
saveFolder = os.path.join(outFolder, code)
if ~os.path.exists(saveFolder):
    os.mkdir(saveFolder)
# ts map
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})

lat, lon = DF.getGeo()
indC = codeLst.index(code)
indS = np.where(~matRm[:, indC])[0]
importlib.reload(figplot)
importlib.reload(axplot)
importlib.reload(mapplot)

yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)

# plot map and scatter
figM = plt.figure(figsize=(16, 3))
gsM = gridspec.GridSpec(1, 5)
axS = figM.add_subplot(gsM[0, :1])
axS.set_title('A) LSTM vs WRTDS')
cs = axplot.scatter121(axS, corrL2[indS, indC],
                       corrW2[indS, indC], matLR[indS, indC])
axS.set_xlabel(r'$R_{LSTM}$')
axS.set_ylabel(r'$R_{WRTDS}$')
plt.colorbar(cs, orientation='vertical', label='linearity')
for ind in [DF.siteNoLst.index(siteNo) for siteNo in siteLst]:
    circle = plt.Circle([corrL2[ind, indC], corrW2[ind, indC]],
                        0.05, color='k', fill=False)
    axS.add_patch(circle)
axM1 = mapplot.mapPoint(
    figM, gsM[0, 1:3], lat[indS], lon[indS], corrL2[indS, indC], s=24,cmap='viridis')
axM1.set_title(r'B) $R_{LSTM}$'+' of {}'.format(codeStr))
axM2 = mapplot.mapPoint(
    figM, gsM[0, 3:], lat[indS], lon[indS], corrL2[indS, indC]**2-corrW2[indS, indC]**2, s=24,
    vRange=[-0.1, 0.1],cmap='viridis')
axM2.set_title(r'C) $\Delta R^2_{LSTM-WRTDS}$'+' of {}'.format(codeStr))
for ind in [DF.siteNoLst.index(siteNo) for siteNo in siteLst]:
    circle = plt.Circle([lon[ind], lat[ind]],
                        2, color='k', fill=False)
    axM1.add_patch(circle)
    circle = plt.Circle([lon[ind], lat[ind]],
                        2, color='k', fill=False)
    axM2.add_patch(circle)
figM.tight_layout()
figM.show()
figM.savefig(os.path.join(saveFolder, 'map_{}'.format(code)))
figM.savefig(os.path.join(saveFolder, 'map_{}.svg'.format(code)))

# plot TS
for siteNo, figN in zip(siteLst, 'DEF'):
    importlib.reload(axplot)
    ind = DF.siteNoLst.index(siteNo)
    dataPlot = [yW[:, ind, indC], yP[:, ind, indC],
                DF.c[:, ind, DF.varC.index(code)]]
    cLst = 'kbr'
    cLst=  ['#377eb8','#e41a1c','k']
    # legLst = [r'WRTDS $\rho$={:.2f}'.format(corrW2[ind, indC]),
    #           r'LSTM $\rho$={:.2f}'.format(corrL2[ind, indC]),
    #           '{} obs'.format(codeStr)]
    legLst = ['WRTDS', 'LSTM', 'Obs.']
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(1, ny, wspace=0)
    axP0 = figP.add_subplot(gsP[0, 0])
    axPLst = [axP0]
    for k in range(1, ny):
        axP = figP.add_subplot(gsP[0, k], sharey=axP0)
        axPLst.append(axP)
    axP = np.array(axPLst)
    axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst)
    for ax in axP:
        ax.set_xlabel('')
        ax.set_xticklabels('')
    titleStr = r'{}) {} of site {} {}={:.2f}; {}={:.2f}'.format(
        figN, codeStr, DF.siteNoLst[ind], '$R_{LSTM}$', corrL2[ind, indC], '$R_{LSTM}$', corrW2[ind, indC])
    figP.suptitle(titleStr)
    figP.tight_layout()
    figP.show()
    figP.savefig(os.path.join(saveFolder, 'tsYr5_{}_{}'.format(code, siteNo)))
    figP.savefig(os.path.join(
        saveFolder, 'tsYr5_{}_{}.svg'.format(code, siteNo)))
