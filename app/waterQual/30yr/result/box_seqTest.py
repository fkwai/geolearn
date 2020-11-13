
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
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)

# single
labelLst = ['QFP_C', 'QTFP_C', 'QT_C']
cLst = 'kgrmbc'
labLst2 = [x.replace('_', '->') for x in labelLst]

# point test
corrMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
for iLab, label in enumerate(labelLst):
    trainSet = 'comb-B10'
    testSet = 'comb-A10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    master = basins.loadMaster(outName)
    yP, ycP = basins.testModel(
        outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
    for iCode, code in enumerate(codeLst):
        ic = wqData.varC.index(code)
        ind = wqData.subset[testSet]
        info = wqData.info.iloc[ind].reset_index()
        ic = wqData.varC.index(code)
        if len(wqData.c.shape) == 3:
            p = yP[-1, :, master['varY'].index(code)]
            o = wqData.c[-1, ind, ic]
        elif len(wqData.c.shape) == 2:
            p = ycP[:, master['varYC'].index(code)]
            o = wqData.c[ind, ic]
        for siteNo in dictSite[code]:
            iS = siteNoLst.index(siteNo)
            indS = info[info['siteNo'] == siteNo].index.values
            rmse, corr = utils.stat.calErr(p[indS], o[indS])
            corrMat[iS, iCode, iLab] = corr
            rmseMat[iS, iCode, iLab] = rmse

# load all sequence
dictLSTMLst = list()
for label in labelLst:
    dictLSTM = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outName, siteNo)
        dictLSTM[siteNo] = df
    dictLSTMLst.append(dictLSTM)
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(siteNo, varLst=codeLst, freq='W')
    dictObs[siteNo] = df
# calculate correlation
corrMat2 = np.full([nSite, len(codeLst), len(labelLst)], np.nan)
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
dictLSTM = dictLSTMLst[1]
for k, label in enumerate(labelLst):
    dictLSTM = dictLSTMLst[k]
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictObs[siteNo][code].iloc[ind2].values
            rmse, corr = utils.stat.calErr(v1, v2)
            corrMat2[indS, ic, k] = corr

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
# labLst2 = ['LSTM vs WRTDS', 'LSTM vs Obs', 'WRTDS vs Obs']
labLst2 = None
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrMat[:, k, i])
        temp.append(corrMat2[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst=cLst,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
