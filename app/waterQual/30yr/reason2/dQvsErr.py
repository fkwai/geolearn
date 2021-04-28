import random
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality as wq
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
if True:
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, codeLst)
    corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)
    # load basin attributes
    dfG = gageII.readData(siteNoLst=siteNoLst)
    dfG = gageII.updateRegion(dfG)
    dfG = gageII.updateCode(dfG)

t = dictObs[siteNoLst[0]].index.values
tt = np.datetime64('2010-01-01')
t0 = np.datetime64('1980-01-01')
ind1 = np.where((t < tt) & (t >= t0))[0]
ind2 = np.where(t >= tt)[0]

# calculate dQ/dt and error
dictErr = dict()
for siteNo in siteNoLst:
    dfObs = dictObs[siteNo]
    dfLSTM = dictLSTM[siteNo]
    area = dfG['DRAIN_SQKM'][siteNo]
    dfDQ = dfObs['00060'].diff()/area
    dfDQ.name = 'dQ'
    dfErr = dfLSTM[codeLst]-dfObs[codeLst]
    dfErr = dfErr.join(dfDQ)
    dictErr[siteNo] = dfErr

codeLst2 = ['00010', '00095', '00300', '00400', '00405',
            '00600', '00605', '00618', '00660', '00665',
            '00681', '00915', '00925', '00930', '00935',
            '00940', '00945', '00955', '71846', '80154']
nfy, nfx = [5, 4]

code = '00915'
xLst = list()
yLst = list()
for siteNo in siteNoLst:
    dfErr = dictErr[siteNo]
    x = dfErr['dQ']
    y = dfErr[code].values
    [xx, yy] = utils.rmNan([x[ind2], y[ind2]], returnInd=False)
    xLst.append(xx)
    yLst.append(yy)
xMat = np.concatenate(xLst)
yMat = np.concatenate(yLst)
fig, ax = plt.subplots(1, 1)
ax.plot(xMat, yMat, '*')
ax.set_xlabel('dQ/dt')
ax.set_ylabel('error')
fig.show()

siteNoCode = dictSite[code]
siteNo = random.choice(siteNoCode)
dfErr = dictErr[siteNo]
x = dfErr['dQ']
y = dfErr[code].values
[xx, yy] = utils.rmNan([x[ind2], y[ind2]], returnInd=False)
fig, ax = plt.subplots(1, 1)
ax.plot(xx, yy, '*')
fig.show()


# attr vs diff
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    xLst = list()
    yLst = list()
    for siteNo in siteNoLst:
        dfErr = dictErr[siteNo]
        x = dfErr['dQ']
        y = dfErr[code].values
        [xx, yy] = utils.rmNan([x[ind2], y[ind2]], returnInd=False)
        xLst.append(xx)
        yLst.append(yy)
    xMat = np.concatenate(xLst)
    yMat = np.concatenate(yLst)
    ax.plot(xMat, yMat, '*')
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()


indC = [codeLst.index(code) for code in codeLst2]
labelLst = ['{} {}'.format(code, usgs.codePdf.loc[code]['shortName'])
            for code in codeLst2]
xMat = corrMat[:, indC, 1]
yMat = corrMat[:, indC, 2]
nXY = [nfx, nfy]


def funcM():
    figM, axM = figplot.scatter121Batch(
        xMat, yMat, cMat, labelLst, nXY, optCb=1, cR=cR,
        ticks=[-0.5, 0, 0.5, 1])
    figM.suptitle('corr of LSTM vs WRTDS colored by {}'.format(cVar))
    figP = plt.figure(figsize=[12, 6])
    gsP = gridspec.GridSpec(3, 3)
    axP = list()
    axP.append(figP.add_subplot(gsP[0, 0]))
    axP.append(figP.add_subplot(gsP[0, 1]))
    axP.append(figP.add_subplot(gsP[0, 2]))
    axP.append(figP.add_subplot(gsP[1, :2]))
    axP.append(figP.add_subplot(gsP[2, :2]))
    axP.append(figP.add_subplot(gsP[1:, 2]))
    return figM, axM, figP, axP, xMat, yMat, labelLst


def funcP(axP, iP, iM):
    xx = xMat[:, iM]
    yy = yMat[:, iM]
    cc = cMat[:, iM] if cMat.ndim == 2 else cMat
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    # maps
    axplot.mapPoint(axP[0], lat, lon, xx, vRange=[-0.5, 1], s=16, cb=False)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[0].add_patch(circle)
    axplot.mapPoint(axP[1], lat, lon, yy, vRange=[-0.5, 1], s=16, cb=False)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[1].add_patch(circle)
    axplot.mapPoint(axP[2], lat, lon, cc, vRange=cR, s=16, cb=False)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[2].add_patch(circle)
    siteNo = siteNoLst[iP]
    # ts
    code = codeLst2[iM]
    print(code, siteNo)
    print(iP, iM)
    v0 = dictObs[siteNo][code].values
    v1 = dictLSTM[siteNo][code].values
    v2 = dictWRTDS[siteNo][code].values
    t = dictObs[siteNo].index.values
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP[3], t[ind1], [v1[ind1], v2[ind1], v0[ind1]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    axplot.plotTS(axP[4], t[ind2], [v1[ind2], v2[ind2], v0[ind2]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    # cq
    q = dictObs[siteNo]['00060'].values
    c = dictObs[siteNo][code].values
    td = dictObs[siteNo].index.dayofyear
    sc = axP[5].scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)


def funcT(iP, iM):
    cc = cMat[iP, iM] if cMat.ndim == 2 else cMat[iP]
    title = '{}; siteNo {} \n corrLSTM {:.2f}; corrWRTDS {:.2f}; {} {}'.format(
        labelLst[iM], siteNoLst[iP], xMat[iP, iM], yMat[iP, iM], cVar, cc)
    return title


importlib.reload(figplot)
figM, figP = figplot.clickMulti(funcM, funcP, funcT=funcT)
