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
if False:
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

# plot 121
code = '00930'
ic = codeLst.index(code)
xMat = corrMat[:, ic, 1]
yMat = corrMat[:, ic, 2]


def funcM():
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    lat[np.isnan(xMat)] = 9999
    lon[np.isnan(xMat)] = 9999
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, xMat**2-yMat**2, vRange=[-0.3, 0.3], s=16)
    axM.set_title('testing Rsq LSTM - Rsq WRTDS')
    figP = plt.figure(figsize=[16, 6])
    axP = list()
    gsP = gridspec.GridSpec(2, 3)
    axP.append(figP.add_subplot(gsP[0, :2]))
    axP.append(figP.add_subplot(gsP[1, :2]))
    axP.append(figP.add_subplot(gsP[0:, 2]))
    axP = np.array(axP)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    figP.suptitle('siteNo {} corrLSTM {:.2f}; corrWRTDS {:.2f}'.format(
        siteNoLst[iP], xMat[iP], yMat[iP]))
    siteNo = siteNoLst[iP]
    # ts
    v0 = dictObs[siteNo][code].values
    v1 = dictLSTM[siteNo][code].values
    v2 = dictWRTDS[siteNo][code].values
    t = dictObs[siteNo].index.values
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP[0], t[ind1], [v1[ind1], v2[ind1], v0[ind1]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    axplot.plotTS(axP[1], t[ind2], [v1[ind2], v2[ind2], v0[ind2]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    # cq
    q = dictObs[siteNo]['00060'].values
    c = dictObs[siteNo][code].values
    td = dictObs[siteNo].index.dayofyear
    sc = axP[2].scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)


importlib.reload(axplot)
figM, figP = figplot.clickMap(funcM, funcP)
