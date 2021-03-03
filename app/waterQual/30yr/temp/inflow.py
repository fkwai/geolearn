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

code = '00915'
# siteNoTemp = ['07104905', '07103700', '07105500', '07105530']
siteNoTemp = ['07105500', '07105530']
figP, axP = plt.subplots(2, 1)
for k, siteNo in enumerate(siteNoTemp):
    v0 = dictObs[siteNo][code].values
    v1 = dictLSTM[siteNo][code].values
    v2 = dictWRTDS[siteNo][code].values
    t = dictObs[siteNo].index.values
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP[k], t, [v1, v2, v0], tBar=np.datetime64('2010-01-01'),
                  styLst='--*', cLst='rbk', legLst=legLst)
    ic = codeLst.index(code)
    indS = siteNoLst.index(siteNo)
    axP[k].set_title('{} LSTM={:.3f} WRTDS={:.3f}'.format(
        siteNo, corrMat[indS, ic, 1], corrMat[indS, ic, 2]))
    axP[k].set_ylim([0, 120])
    if k != len(siteNoTemp)-1:
        axP[k].set_xticklabels([])
figP.show()

siteNoTemp = ['07105500', '07105530']
figP, axP = plt.subplots(1, 2)
for k, siteNo in enumerate(siteNoTemp):
    q = dictObs[siteNo]['00060'].values
    c = dictObs[siteNo][code].values
    td = dictObs[siteNo].index.dayofyear
    sc = axP[k].scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
    axP[k].set_title('{}'.format(
        siteNo))
figP.show()


siteNoTemp = ['07105500', '07105530']
df1 = dictObs[siteNoTemp[0]][code]
df2 = dictObs[siteNoTemp[1]][code]
df = df2-df1
figP, axP = plt.subplots(1, 1)
axplot.plotTS(axP, df.index, [df1.values, df2.values], tBar=np.datetime64('2010-01-01'),
              styLst='**', cLst='rb', legLst=siteNoTemp)
figP.show()

siteNoTemp = ['07105500', '07105530']
df1 = dictLSTM[siteNoTemp[0]][code]
df2 = dictLSTM[siteNoTemp[1]][code]
df = df2-df1
figP, axP = plt.subplots(1, 1)
axplot.plotTS(axP, df.index, [df1.values, df2.values], tBar=np.datetime64('2010-01-01'),
              cLst='rb', legLst=siteNoTemp)
figP.show()


siteNoTemp = ['07105500', '07105530']
df1 = dictObs[siteNoTemp[0]]['00060']
df2 = dictObs[siteNoTemp[1]]['00060']
df = df2-df1
figP, axP = plt.subplots(1, 1)
axplot.plotTS(axP, df.index, [df1.values, df2.values], tBar=np.datetime64('2010-01-01'),
              styLst='--', cLst='rb', legLst=siteNoTemp)
figP.show()

siteNoTemp = ['07105500', '07105530']
v1 = dictObs[siteNoTemp[0]]['00060'].values * \
    dictObs[siteNoTemp[0]][code].values
v2 = dictObs[siteNoTemp[1]]['00060'].values * \
    dictObs[siteNoTemp[1]][code].values
t = dictObs[siteNoTemp[0]].index.values
figP, axP = plt.subplots(1, 1)
[vv1, vv2], indT = utils.rmNan([v1, v2])
axplot.plotTS(axP, t[indT], [vv1- vv2], tBar=np.datetime64('2010-01-01'),
              cLst='rb')
figP.show()

siteNoTemp = ['07105500', '07105530']
v1 = dictObs[siteNoTemp[0]][code].values
v2 = dictObs[siteNoTemp[1]][code].values
t = dictObs[siteNoTemp[0]].index.values
figP, axP = plt.subplots(1, 1)
[vv1, vv2], indT = utils.rmNan([v1, v2])
axplot.plotTS(axP, t[indT], [vv1, vv2], tBar=np.datetime64('2010-01-01'),
              cLst='rb', legLst=siteNoTemp)
figP.show()

siteNoTemp = ['07105500', '07105530']
df1 = dictWRTDS[siteNoTemp[0]][code]
df2 = dictWRTDS[siteNoTemp[1]][code]
df = df2-df1
figP, axP = plt.subplots(1, 1)
axplot.plotTS(axP, df.index, [df1.values, df2.values], tBar=np.datetime64('2010-01-01'),
              cLst='rb', legLst=siteNoTemp, styLst='--')
figP.show()
