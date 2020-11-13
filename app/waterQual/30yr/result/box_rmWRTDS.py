
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

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
codeLst = sorted(usgs.newC)
ep = 500
reTest = False
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
dataLst = 'rbWN5-WRTDS'

# single
label = 'QTFP_C'

# load all sequence
dictLSTMLst = list()
for dataName in ['rbWN5', 'rbWN5-WRTDS']:
    dictLSTM = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outName, siteNo)
        dictLSTM[siteNo] = df
    dictLSTMLst.append(dictLSTM)
dictWRTDS = dict()
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10', 'output')
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None)
    df = utils.time.datePdf(df)
    dictWRTDS[siteNo] = df
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(siteNo, varLst=codeLst, freq='W')
    dictObs[siteNo] = df

# calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
dictLSTM = dictLSTMLst[1]
corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v0 = dictLSTM[siteNo][code+'-R'].iloc[ind2].values
        v2 = dictWRTDS[siteNo][code].iloc[ind2].values
        v3 = dictObs[siteNo][code].iloc[ind2].values
        v1 = v0+v2
        rmse1, corr1 = utils.stat.calErr(v1, v2)
        rmse2, corr2 = utils.stat.calErr(v1, v3)
        rmse3, corr3 = utils.stat.calErr(v2, v3)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['LSTM vs WRTDS', 'LSTM vs Obs', 'WRTDS vs Obs']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='gbr',
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
