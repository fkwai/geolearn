
import matplotlib
# from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
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
            siteNo, varLst=['00060']+codeLst, freq='W')
        dictObs[siteNo] = df

    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    indT1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
    indT2 = np.where(df.index.values >= tt)[0]
    dictLSTM = dictLSTMLst[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    rmseMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[indT2].values
            v2 = dictWRTDS[siteNo][code].iloc[indT2].values
            v3 = dictObs[siteNo][code].iloc[indT2].values
            vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
            rmse1, corr1 = utils.stat.calErr(vv1, vv2)
            rmse2, corr2 = utils.stat.calErr(vv1, vv3)
            rmse3, corr3 = utils.stat.calErr(vv2, vv3)
            corrMat[indS, ic, 0] = corr1
            corrMat[indS, ic, 1] = corr2
            corrMat[indS, ic, 2] = corr3
            rmseMat[indS, ic, 0] = rmse1
            rmseMat[indS, ic, 1] = rmse2
            rmseMat[indS, ic, 2] = rmse3

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

    # calculate LombScargle
    pMat = np.full([len(siteNoLst), len(codeLst)], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            df = dictObs[siteNo]
            t = np.arange(len(df))*7
            y = df[code]
            tt, yy = utils.rmNan([t, y], returnInd=False)
            p = LombScargle(tt, yy).power(1/365)
            pMat[indS, ic] = p

    # calculate CQ relationship
    rMat = np.full([len(siteNoLst), len(codeLst)], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            q = dictObs[siteNo]['00060'].values
            c = dictObs[siteNo][code].values
            qq, cc = utils.rmNan([q, c], returnInd=False)
            corr = np.corrcoef(qq, cc)[1, 0]
            rMat[indS, ic] = corr**2


# matplotlib.rcParams.update({'font.size': 18})
# matplotlib.rcParams.update({'lines.linewidth': 2})
# matplotlib.rcParams.update({'lines.markersize': 12})

# ts map
codeM = ['00660', '00665']
siteNoCode = list(set(dictSite['00915']).intersection(set(dictSite['00955'])))
indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']
cMat = dfG['PCT_NO_ORDER'].values[indS]


def funcM():
    figM = plt.figure(figsize=[12, 3])
    gsM = gridspec.GridSpec(1, 5)
    axM = list()
    axM.append(figM.add_subplot(gsM[0, 0]))
    axM.append(figM.add_subplot(gsM[0, 1:3]))
    axM.append(figM.add_subplot(gsM[0, 3:5]))
    axM = np.array(axM)g
    x = corrMat[indS, codeLst.index(codeM[0]), 1]
    y = corrMat[indS, codeLst.index(codeM[1]), 1]
    sc = axplot.scatter121(axM[0], x, y, cMat)
    figM.colorbar(sc, ax=axM[0])
    axM[0].set_label('scatter')
    axplot.mapPoint(axM[1], lat, lon, x, s=24)
    axM[1].set_label('map1')
    axplot.mapPoint(axM[2], lat, lon, y, s=24)
    axM[2].set_label('map2')
    labelLst = ['scatter', 'map1', 'map2']
    xMat = np.stack([x, lon, lon], axis=1)
    yMat = np.stack([y, lat, lat], axis=1)

    figP = plt.figure(figsize=[12, 3])
    gsP = gridspec.GridSpec(2, 4)
    axP = list()
    axP.append(figP.add_subplot(gsP[0, :-1]))
    axP.append(figP.add_subplot(gsP[0, -1]))
    axP.append(figP.add_subplot(gsP[1, :-1]))
    axP.append(figP.add_subplot(gsP[1, -1]))
    axP = np.array(axP)

    return figM, axM, figP, axP, xMat, yMat, labelLst


def funcP(axP, iP, iM):
    siteNo = siteNoCode[iP]
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    for k, code in enumerate(codeM):
        v1 = dictLSTM[siteNo][code].values
        v2 = dictWRTDS[siteNo][code].values
        q = dictObs[siteNo]['00060'].values
        o = dictObs[siteNo][code].values
        t = dictObs[siteNo].index.values
        iS = siteNoLst.index(siteNo)
        ic = codeLst.index(code)
        legLst = ['LSTM', 'WRTDS', 'Obs']
        axplot.plotTS(axP[k*2], t, [v1, v2, o], tBar=tBar, sd=sd,
                      styLst='--*', cLst='rbk', legLst=legLst)
        axP[k*2].set_title('site {} corrLSTM={:.2f} corrWRTDS={:.2f}'.format(
            siteNo, corrMat[iS, ic, 1], corrMat[iS, ic, 2]))
        td = dictObs[siteNo].index.dayofyear
        sc = axP[k*2+1].scatter(np.log(q), o, c=td,
                                cmap='hsv', vmin=0, vmax=365)


importlib.reload(figplot)
figM, figP = figplot.clickMulti(funcM, funcP)

th = 0.75
temp = np.where((corrMat[indS, ic, 1] > th) & (corrMat[indS, ic, 2] > th))[0]
len(temp)
len(indS)
