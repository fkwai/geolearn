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
    dictLSTMLst = list()
    # LSTM
    labelLst = ['QTFP_C']
    for label in labelLst:
        dictLSTM = dict()
        trainSet = 'comb-B10'
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        for k, siteNo in enumerate(siteNoLst):
            print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
            df = basins.loadSeq(outName, siteNo)
            dictLSTM[siteNo] = df
        dictLSTMLst.append(dictLSTM)
    # WRTDS
    dictWRTDS = dict()
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-W', 'B10', 'output')
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirWRTDS, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        # df = utils.time.datePdf(df)
        dictWRTDS[siteNo] = df
    # Observation
    dictObs = dict()
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+gridMET.varLst+codeLst, freq='W')
        dictObs[siteNo] = df

    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    ind1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
    ind2 = np.where(df.index.values >= tt)[0]
    dictLSTM = dictLSTMLst[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictWRTDS[siteNo][code].iloc[ind2].values
            v3 = dictObs[siteNo][code].iloc[ind2].values
            vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
            rmse1, corr1 = utils.stat.calErr(vv1, vv2)
            rmse2, corr2 = utils.stat.calErr(vv1, vv3)
            rmse3, corr3 = utils.stat.calErr(vv2, vv3)
            corrMat[indS, ic, 0] = corr1
            corrMat[indS, ic, 1] = corr2
            corrMat[indS, ic, 2] = corr3

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
# cMat = np.argmax(matV, axis=1)
# cR = [0, 15]
cVar = 'GLiM'
cMat=matV[:,5]
cR=[0,1]
# cMat = np.log(dfG['DRAIN_SQKM'].values)
# cR = None

tabGlim.loc['05465500']

# plot 121
plt.close('all')
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [5, 4]

# codeLst2 = ['00095', '00618', '00915', '00925', '00935', '00955']
# nfy, nfx = [3, 2]

codeLst2 = ['00915', '00955']
nfy, nfx = [2, 1]

# attr vs diff
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = cMat
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
    axplot.mapPoint(axP[0], lat, lon, xx, vRange=[-0.5, 1], s=16)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[0].add_patch(circle)
    axplot.mapPoint(axP[1], lat, lon, yy, vRange=[-0.5, 1], s=16)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[1].add_patch(circle)
    axplot.mapPoint(axP[2], lat, lon, cc, vRange=cR, s=16)
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


importlib.reload(figplot)
figM, figP = figplot.clickMulti(funcM, funcP, funcT=funcT)
