
import importlib
import numpy as np
import os
import pandas as pd
import json
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import usgs, gageII
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
codeLst = sorted(usgs.newC)

# load Linear and Seasonal model
dictL = dict()
dirL = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-WL', 'All', 'output')
dictS = dict()
dirS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-WS', 'All', 'output')
for dirTemp, dictTemp in zip([dirL, dirS], [dictL, dictS]):
    for k, siteNo in enumerate(siteNoLst):
        print('\t WRTDS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirTemp, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        dictTemp[siteNo] = df

dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t USGS site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(
        siteNo, varLst=['00060']+codeLst, freq='W', rmFlag=True)
    dictObs[siteNo] = df

# calculate correlation
corrMatTemp = np.full([len(siteNoLst), len(codeLst), 2], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictL[siteNo][code].values
        v2 = dictS[siteNo][code].values
        v0 = dictObs[siteNo][code].values
        (vv0, vv1, vv2), indV = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(vv1, vv0)
        rmse2, corr2 = utils.stat.calErr(vv2, vv0)
        corrMatTemp[indS, ic, 0] = corr1
        corrMatTemp[indS, ic, 1] = corr2

rMat = corrMatTemp**2
codeLst2 = ['00915', '00925', '00930', '00935', '00940', '00945',
            '00955', '70303', '80154']
[nfy, nfx] = [3, 3]

codeLst2 = ['00010', '00300', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [4, 5]

fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = rMat[:, ic, 1]
    y = rMat[:, ic, 0]
    axplot.plot121(ax, x, y, vR=[0, 1])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateRegion(dfG)
dfG = gageII.updateCode(dfG)

fileGlim = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'tab_1KM')
tabGlim = pd.read_csv(fileGlim, dtype={'siteNo': str}).set_index('siteNo')
matV = np.argmax(tabGlim.values, axis=1)

labelLst = ['{} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
            for code in codeLst2]
icLst = [codeLst.index(code) for code in codeLst2]
figM, axM = figplot.scatter121Batch(
    rMat[:, icLst, 1], rMat[:, icLst, 0], matV, labelLst, [nfx, nfy], optCb=1,
    ticks=[0, 0.5, 1])
figM.show()



temp = ['00930', '00940']
ic1 = codeLst.index(temp[0])
ic2 = codeLst.index(temp[1])
nameLst = [usgs.codePdf.loc[code]['shortName'] for code in temp]
fig, axes = plt.subplots(1, 2)
axplot.scatter121(axes[0], rMat[:, ic1, 0], rMat[:, ic2, 0], qMat, vR=[0, .6])
axes[0].set_xlabel('Linearity of {}'.format(nameLst[0]))
axes[0].set_ylabel('Linearity of {}'.format(nameLst[1]))
axplot.scatter121(axes[1], rMat[:, ic1, 1], rMat[:, ic2, 1], qMat, vR=[0, .6])
axes[1].set_xlabel('Seasonality of {}'.format(nameLst[0]))
axes[1].set_ylabel('Seasonality of {}'.format(nameLst[1]))
fig.show()
