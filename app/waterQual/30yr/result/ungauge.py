from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import json


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSiteN5 = json.load(f)
with open(os.path.join(dirSel, 'dictRB_Y30N2.json')) as f:
    dictSiteN2 = json.load(f)
codeLst = sorted(usgs.newC)

dictSite = dict()
for code in usgs.newC+['comb']:
    siteNoCode = list(set(dictSiteN2[code])-set(dictSiteN5['comb']))
    dictSite[code] = siteNoCode
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

corrMat = np.full([nSite, len(codeLst)], np.nan)
rmseMat = np.full([nSite, len(codeLst)], np.nan)

ep = 500
reTest = True
wqData = waterQuality.DataModelWQ('rbWN2')
testSet = 'comb-B10'
label = 'FP_QC'
outName = '{}-{}-{}-{}-ungauge'.format('rbWN5', 'comb', label, testSet)
master = basins.loadMaster(outName)
yP, ycP = basins.testModel(
    outName, testSet, wqData=wqData, ep=ep, reTest=reTest)

dictP = dict()
dictO = dict()
for iCode, code in enumerate(codeLst):
    pLst = list()
    oLst = list()
    ic = wqData.varC.index(code)
    ind = wqData.subset[testSet]
    info = wqData.info.iloc[ind].reset_index()
    ic = wqData.varC.index(code)
    p = ycP[:, master['varYC'].index(code)]
    o = wqData.c[ind, ic]
    for siteNo in dictSite[code]:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iCode] = corr
        rmseMat[iS, iCode] = rmse
        pLst.append(p[indS])
        oLst.append(o[indS])
    dictP[code] = np.concatenate(pLst)
    dictO[code] = np.concatenate(oLst)


fig, ax = plt.subplots(1, 1)
ax.plot(dictP['00955'], dictO['00955'],'*')
fig.show()

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    temp.append(corrMat[:, k])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      figsize=(12, 4), yRange=[0, 1])
fig.show()
