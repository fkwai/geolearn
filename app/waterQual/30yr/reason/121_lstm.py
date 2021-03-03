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
if True:
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
        df,_ = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+codeLst, freq='W')
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

# color mat - region
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

cMat = rMat
cVar = 'power'

cVar = 'NWIS_DRAIN_SQKM'
cMat = dfG[cVar].values


# plot 121
# plt.close('all')
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [5, 4]

# codeLst2 = ['00010', '00300']
# nfy, nfx = [2, 1]

# attr vs diff
for cMat in [rMat, pMat]:
    fig, axes = plt.subplots(nfy, nfx)
    for k, code in enumerate(codeLst2):
        j, i = utils.index2d(k, nfy, nfx)
        ax = axes[j, i]
        ic = codeLst.index(code)
        if cMat.ndim > 1:
            x = cMat[:, ic]
        else:
            x = cMat
        y = corrMat[:, ic, 2]
        ax.set_ylim(0, 1)
        ax.plot(x, y, '*')
        titleStr = '{} {} '.format(
            code, usgs.codePdf.loc[code]['shortName'])
        ax.set_title(titleStr)
    fig.show()
