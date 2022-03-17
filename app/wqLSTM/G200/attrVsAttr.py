
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
from hydroDL.app.waterQuality import WRTDS

import warnings
# warnings.simplefilter('error')

dataName = 'G200N'

# with warnings.catch_warnings():
#     warnings.simplefilter('ignore', category=RuntimeWarning)
#     DF = dbBasin.DataFrameBasin(dataName)
DF = dbBasin.DataFrameBasin(dataName)

codeLst = usgs.newC

trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

trainSet = 'rmR20'
testSet = 'pkR20'
# trainSet = 'B10'
# testSet = 'A10'
labelLst = ['QFPRT2C', 'QFRT2C', 'QFPT2C', 'FPRT2C']
nL = len(labelLst)
yLst = list()
for label in labelLst:
    outName = '{}-{}-{}'.format(dataName, label, trainSet)
    yP, ycP = basinFull.testModel(
        outName, DF=DF, testSet=testSet, ep=500)
    yOut = np.ndarray(yP.shape)
    for k, code in enumerate(codeLst):
        m = DF.g[:, DF.varG.index(code+'-M')]
        s = DF.g[:, DF.varG.index(code+'-S')]
        yOut[:, :, k] = yP[:, :, k]*s+m
    yLst.append(yOut)


# WRTDS
# yW = WRTDS.testWRTDS(dataName, trainSet, testSet, codeLst)
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation matrix
d1 = dbBasin.DataModelBasin(DF, subset=trainSet, varY=codeLst)
d2 = dbBasin.DataModelBasin(DF, subset=testSet, varY=codeLst)
siteNoLst = DF.siteNoLst
matW = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
matLst = [np.full([len(siteNoLst), len(codeLst), 4], np.nan) for x in labelLst]

for indS, siteNo in enumerate(siteNoLst):
    print(indS)
    for indC, code in enumerate(codeLst):
        n1 = np.sum(~np.isnan(d1.Y[:, indS, indC]), axis=0)
        n2 = np.sum(~np.isnan(d2.Y[:, indS, indC]), axis=0)
        if n1 >= 160 and n2 >= 40:
            statW = utils.stat.calStat(yW[:, indS, indC], d2.Y[:, indS, indC])
            matW[indS, indC, :] = list(statW.values())
            for k in range(nL):
                yL = yLst[k]
                statL = utils.stat.calStat(
                    yL[:, indS, indC], d2.Y[:, indS, indC])
                matLst[k][indS, indC, :] = list(statL.values())

# load basin attributes
regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM',
             'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
dfG = gageII.readData(siteNoLst=siteNoLst)
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
dfG = gageII.updateCode(dfG)


# varLst = ['PLANTNLCD06', 'NITR_APP_KG_SQKM']
# codePlot = ['00600', '00618']
varLst = ['SNOW_PCT_PRECIP', 'STREAMS_KM_SQ_KM']
codePlot = codeLst
fig, axes = plt.subplots(4, 5)
x = dfG[varLst[0]].values
y = dfG[varLst[1]].values
for k, code in enumerate(codePlot):
    j, i = utils.index2d(k, 4, 5)
    ax = axes[j, i]
    ic = codeLst.index(code)
    # c = matLst[2][:, ic, 3]**2-matLst[0][:, ic, 3]**2
    c = matLst[0][:, ic, 3]**2-matW[:, ic, 3]**2
    sc = ax.scatter(x, y, c=c, vmin=-0.3, vmax=0.3, cmap='jet')
    ax.set_xlabel(varLst[0])
    ax.set_ylabel(varLst[1])
    ax.set_title('Rsq increment of {} {}'.format(
        code, usgs.codePdf.loc[code]['shortName']))
    fig.colorbar(sc, ax=ax)
fig.show()

var = 'PCT_1ST_ORDER'
codePlot = codeLst
fig, axes = plt.subplots(4, 5)
x = dfG[var].values
for k, code in enumerate(codePlot):
    j, i = utils.index2d(k, 4, 5)
    ax = axes[j, i]
    ic = codeLst.index(code)
    # c = matLst[2][:, ic, 3]**2-matLst[0][:, ic, 3]**2
    c = matLst[0][:, ic, 3]**2-matW[:, ic, 3]**2
    sc = ax.plot(x, c, '*')
    ax.set_xlabel(var)
    titleStr = '{} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
    ax.axhline(0, color='r')
    axplot.titleInner(ax, titleStr)
fig.show()


'corr diff of {} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
