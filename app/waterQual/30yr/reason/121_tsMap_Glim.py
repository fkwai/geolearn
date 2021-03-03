import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
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
if False:
    importlib.reload(wq.wqLoad)
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, codeLst)
    corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)
    # load basin attributes
    dfG = gageII.readData(siteNoLst=siteNoLst)
    dfG = gageII.updateRegion(dfG)
    dfG = gageII.updateCode(dfG)

# playground !
# estimate travel time
d = dfG['ROCKDEPAVE'].values  # inches
a = dfG['DRAIN_SQKM'].values  # sqkm
c = dfG['AWCAVE'].values  # []
q = np.ndarray(len(siteNoLst))
for k, siteNo in enumerate(siteNoLst):
    q[k] = dictObs[siteNo]['00060'].mean()  # cubic feet / s
unitCov = 0.0254*10**6/0.3048**3/24/60/60/365  # year

# GLiM
fileGlim = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'tab_1KM')
tabGlim = pd.read_csv(fileGlim, dtype={'siteNo': str}).set_index('siteNo')
matV = tabGlim.values
cMat = np.argmax(matV, axis=1)
cR = [0, 15]
cVar = 'GLiM'
cMat = matV[:, 4]+matV[:, 5]
cR = [0, 0.5]
# cMat = np.log(dfG['DRAIN_SQKM'].values)
# cR = None

# plot 121
plt.close('all')
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [5, 4]

codeLst2 = ['00915', '00925']
nfy, nfx = [2, 1]

combineLst = [
    [1, 2],
    [4, 5],
    [1, 2, 4],
    [7, 8, 10],
    [7, 8, 10, 11],
    [9, 12]
]
# attr vs diff
for kk in range(6):
    fig, axes = plt.subplots(nfy, nfx)
    for k, code in enumerate(codeLst2):
        j, i = utils.index2d(k, nfy, nfx)
        ax = axes[j]
        ic = codeLst.index(code)
        x = np.sum(matV[:, combineLst[kk]], axis=1)
        y = corrMat[:, ic, 1]**2-corrMat[:, ic, 2]**2
        ax.plot(x, y, '*')
        ax.plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
        ax.set_ylim([-0.5, 0.5])
        # ax.set_xlim(cR)
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
ind1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
ind2 = np.where(df.index.values >= tt)[0]


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
    cc = cMat
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
    title = '{}; siteNo {} \n corrLSTM {:.2f}; corrWRTDS {:.2f}; {} {}'.format(
        labelLst[iM], siteNoLst[iP], xMat[iP, iM], yMat[iP, iM], cVar, cMat[iP])
    return title
