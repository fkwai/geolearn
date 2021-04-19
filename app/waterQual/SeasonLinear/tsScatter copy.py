
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
dirL = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DL', 'All', 'output')
dictS = dict()
dirS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DS', 'All', 'output')
dictQ = dict()
dirQ = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-DQ', 'All', 'output')
for dirTemp, dictTemp in zip([dirL, dirS, dirQ], [dictL, dictS, dictQ]):
    for k, siteNo in enumerate(siteNoLst):
        print('\t WRTDS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirTemp, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        dictTemp[siteNo] = df

dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t USGS site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(
        siteNo, varLst=['00060']+codeLst, freq='D', rmFlag=True)
    dictObs[siteNo] = df

# calculate rsq
rMat = np.full([len(siteNoLst), len(codeLst), 2], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictL[siteNo][code].values
        v2 = dictS[siteNo][code].values
        v0 = dictObs[siteNo][code].values
        (vv0, vv1, vv2), indV = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(vv1, vv0)
        rmse2, corr2 = utils.stat.calErr(vv2, vv0)
        rMat[indS, ic, 0] = corr1**2
        rMat[indS, ic, 1] = corr2**2
qMat = np.full([len(siteNoLst)], np.nan)
for siteNo in siteNoLst:
    indS = siteNoLst.index(siteNo)
    v1 = dictQ[siteNo]['00060'].values
    v0 = dictObs[siteNo]['00060'].values
    (vv0, vv1), indV = utils.rmNan([v0, v1])
    rmse, corr = utils.stat.calErr(vv1, vv0)
    qMat[indS] = corr**2

codeLst2 = ['00915', '00925', '00930', '00935', '00940', '00945',
            '00955', '70303', '80154']
[nfy, nfx] = [3, 3]

# codeLst2 = ['00010', '00300', '00405', '00600', '00605',
#             '00618', '00660', '00665', '00681', '00915',
#             '00925', '00930', '00935', '00940', '00945',
#             '00950', '00955', '70303', '71846', '80154']
# nfy, nfx = [4, 5]

fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = qMat
    y = rMat[:, ic, 0]
    axplot.plot121(ax, x, y, vR=[0, 1])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

cMat = qMat
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = rMat[:, ic, 1]
    y = rMat[:, ic, 0]
    axplot.scatter121(ax, x, y, qMat, vR=[0, 0.5])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

indC = [codeLst.index(code) for code in codeLst2]
labelLst = ['{} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
            for code in codeLst2]
xMat = rMat[:, indC, 1]
yMat = rMat[:, indC, 0]
nXY = [nfx, nfy]
figM, axM = figplot.scatter121Batch(
    xMat, yMat, qMat, labelLst, nXY, optCb=1, cR=[0, 0.6],
    ticks=[0, 0.5, 1], s=20)
figM.show()

ic1 = codeLst.index('00915')
ic2 = codeLst.index('00955')
fig, axes = plt.subplots(1, 2)
# axplot.plot121(axes[0], rMat[:, ic1, 0], rMat[:, ic2, 0], vR=[0, 1])
# axplot.plot121(axes[1], rMat[:, ic1, 1], rMat[:, ic2, 1], vR=[0, 1])
axplot.scatter121(axes[0], rMat[:, ic1, 0], rMat[:, ic2, 0], qMat,  vR=[0, .5])
axplot.scatter121(axes[1], rMat[:, ic1, 1], rMat[:, ic2, 1], qMat,  vR=[0, .5])
fig.show()

temp = ['00915', '00955']
ic1 = codeLst.index(temp[0])
ic2 = codeLst.index(temp[1])
nameLst = [usgs.codePdf.loc[code]['shortName'] for code in temp]
fig, axes = plt.subplots(1, 2)
sc1 = axplot.scatter121(axes[0], rMat[:, ic1, 1],
                        rMat[:, ic1, 0], qMat, vR=[0, 0.5])
axes[0].set_xlabel('Seasonality of {}'.format(nameLst[0]))
axes[0].set_ylabel('Linearity of {}'.format(nameLst[0]))
sc2 = axplot.scatter121(axes[1], rMat[:, ic2, 1],
                        rMat[:, ic2, 0], qMat, vR=[0, 0.5])
axes[1].set_xlabel('Seasonality of {}'.format(nameLst[1]))
axes[1].set_ylabel('Linearity of {}'.format(nameLst[1]))
fig.show()


temp = ['00915', '00955']
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

temp = ['00915', '00955']
ic1 = codeLst.index(temp[0])
ic2 = codeLst.index(temp[1])
nameLst = [usgs.codePdf.loc[code]['shortName'] for code in temp]
fig, axes = plt.subplots(1, 2)
axplot.scatter121(axes[0], qMat, rMat[:, ic1, 1], rMat[:, ic1, 0], vR=[0, 1])
axes[0].set_xlabel('Seasonality of Q')
axes[0].set_ylabel('Seasonality of {}'.format(nameLst[0]))
axplot.scatter121(axes[1], qMat, rMat[:, ic2, 1],  rMat[:, ic2, 0], vR=[0, 1])
axes[1].set_xlabel('Seasonality of Q')
axes[1].set_ylabel('Seasonality of {}'.format(nameLst[1]))
fig.show()

# rock type - Na Cl
fileGlim = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'tab_1KM')
tabGlim = pd.read_csv(fileGlim, dtype={'siteNo': str}).set_index('siteNo')
matV = np.argmax(tabGlim.values, axis=1)
matV = tabGlim.values[:,2]
temp = ['00915', '00955']
ic1 = codeLst.index(temp[0])
ic2 = codeLst.index(temp[1])
nameLst = [usgs.codePdf.loc[code]['shortName'] for code in temp]
fig, axes = plt.subplots(1, 2)
cb = axplot.scatter121(axes[0], rMat[:, ic1, 0],
                       rMat[:, ic2, 0], matV, cmap='jet')
axes[0].set_xlabel('Linearity of {}'.format(nameLst[0]))
axes[0].set_ylabel('Linearity of {}'.format(nameLst[1]))
fig.colorbar(cb, ax=axes[0])
cb = axplot.scatter121(axes[1], rMat[:, ic1, 1],
                       rMat[:, ic2, 1], matV, cmap='jet')
axes[1].set_xlabel('Seasonality of {}'.format(nameLst[0]))
axes[1].set_ylabel('Seasonality of {}'.format(nameLst[1]))
fig.colorbar(cb, ax=axes[1])
fig.show()
