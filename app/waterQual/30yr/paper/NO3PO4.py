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


# code = '00618'
# cVar = 'RIP800_PLANT'
# th = 24.52499962
code = '00660'
cVar = 'PHOS_APP_KG_SQKM'
th = 1209.950012
cMat = dfG[cVar].values
figName = '{}_{}'.format(code, cVar)
dirFig = r'C:\Users\geofk\work\paper\waterQuality'


# attr vs diff
cMatLog = np.log(cMat+1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ic = codeLst.index(code)
sc = axplot.scatter121(axes[0], corrMat[:, ic, 1],
                       corrMat[:, ic, 2], cMat, size=30)
axes[0].set_xlabel('Corr LSTM')
axes[0].set_ylabel('Corr WRTDS')
fig.colorbar(sc, ax=axes[0])
x = cMat
y = corrMat[:, ic, 1]**2-corrMat[:, ic, 2]**2
axes[1].plot(x, y, '*')
axes[1].plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
axes[1].plot([th, th], [-0.5, 0.5], 'r-')
axes[1].set_ylim([-0.5, 0.5])
axes[1].set_xlabel(cVar)
axes[1].set_ylabel('Rsq LSTM - Rsq WRTDS')
fig.suptitle('affect of {} on {} {}'.format(
    cVar, code, usgs.codePdf.loc[code]['shortName']))
fig.show()
fig.savefig(os.path.join(dirFig,figName)+'_a')

# threshold
ind1 = np.where(cMat <= th)
ind2 = np.where(cMat > th)
dataBox = list()
pLst = list()
for ind in [ind1, ind2]:
    a = corrMat[ind, ic, 1].flatten()
    b = corrMat[ind, ic, 2].flatten()
    aa, bb = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    dataBox.append([a, b])
    pLst.append(p)
label1 = ['<={:.3f}\np-value={:.0e}'.format(th, pLst[0]),
          '>{:.3f}\np-value={:.0e}'.format(th, pLst[1])]
fig = figplot.boxPlot(dataBox, label1=label1, label2=['LSTM', 'WRTDS'],
                      widths=0.5, figsize=(6, 4), yRange=[0, 1])
fig.suptitle('affect of {} on {} {}'.format(
    cVar, code, usgs.codePdf.loc[code]['shortName']))
fig.show()
fig.savefig(os.path.join(dirFig,figName)+'_b')