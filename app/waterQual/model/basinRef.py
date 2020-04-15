import sklearn.tree
from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')
figFolder = os.path.join(kPath.dirWQ, 'basinRef')

# compare of opt1-4
outLst = ['basinRef-opt1', 'basinRef-opt2']
trainSet = 'first80'
testSet = 'last20'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData, ep=200)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData, ep=200)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    pLst1.append(p1)
    pLst2.append(p2)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)


codePdf = usgs.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    codeLst = codePdf[codePdf.group == group].index.tolist()
    indLst = [wqData.varC.index(code) for code in codeLst]
    labLst1 = [codePdf.loc[code]['shortName'] +
               '\n'+code for code in codeLst]
    labLst2 = ['opt1-train', 'opt2-train', 'opt1-test', 'opt2-test']
    dataBox = list()
    for ic in indLst:
        temp = list()
        for errMat in errMatLst1:
            temp.append(errMat[:, ic, 1])
        for errMat in errMatLst2:
            temp.append(errMat[:, ic, 1])
        dataBox.append(temp)
    title = 'correlation of {} group'.format(group)
    figName = 'box_{}_allOpt'.format(group)
    fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
    fig.suptitle(title)
    fig.show()
    fig.savefig(os.path.join(figFolder, figName))

siteNoLst = wqData.info['siteNo'].unique().tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'),
                    dtype={'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist() if siteNo in siteNoLst]
dropColLst = ['STANAME', 'WR_REPORT_REMARKS',
              'ADR_CITATION', 'SCREENING_COMMENTS']
dfX = gageII.readData(siteNoLst=siteNoLst).drop(columns=dropColLst)
dfX = gageII.updateCode(dfX)
unitConv = 0.3048**3*365*24*60*60/1000**2


# area vs error
indHBN = [siteNoLst.index(siteNo) for siteNo in siteNoHBN]
area = dfX['DRAIN_SQKM'].values
errMat = errMatLst2[0]
code = '00605'
# code = '00955'
err = errMat[:, wqData.varC.index(code), 1]
fig, ax = plt.subplots(1, 1)
ax.plot(area, err, 'b*')
ax.plot(area[indHBN], err[indHBN], 'r*')
# np.nanmedian(err)
# np.nanmedian(err[indHBN, :])
fig.show()


# dw vs error
code = '00955'
# code = '00600'
pMat = np.full([len(siteNoLst), 2], np.nan)
for k, siteNo in enumerate(siteNoLst):
    area = dfX.loc[siteNo]['DRAIN_SQKM']
    dfC = usgs.readSample(siteNo, codeLst=usgs.codeLst)
    dfQ = usgs.readStreamflow(siteNo)
    df = dfC.join(dfQ)
    t = df.index.values
    q = df['00060_00003'].values/area*unitConv
    c = df[code].values
    try:
        ceq, dw, y = relaCQ.kateModel2(q, c)
        pMat[k, 0] = ceq
        pMat[k, 1] = dw
    except:
        pass
fig, ax = plt.subplots(1, 1)
ax.plot(pMat[:, 1], err, 'b*')
ax.plot(pMat[indHBN, 1], err[indHBN], 'r*')
fig.show()


ind = np.where(~np.isnan(err))[0]
x = dfX.values.astype(float)[ind, :]
y = err[ind]

ind = np.where(~np.isnan(pMat[:, 1]))[0]
x = dfX.values.astype(float)[ind, :]
y = pMat[ind, 1]

x[np.isnan(x)] = -99

featLst = list()
indXLst = list()
thLst = list()
while (len(indXLst) < x.shape[1]-1):
    indInput = list(set(range(x.shape[1]))-set(indXLst))
    clf = sklearn.tree.DecisionTreeRegressor(
        max_depth=1, min_weight_fraction_leaf=0.1)
    clf = clf.fit(x[:, indInput], y)
    tree = clf.tree_
    indX = indInput[tree.feature[0]]
    feat = dfX.columns[indX]
    th = tree.threshold[0]
    featLst.append(feat)
    indXLst.append(indX)
    thLst.append(th)

k = 1
# k = featLst.index('NITR_APP_KG_SQKM')
feat = featLst[k]
th = thLst[k]
fig, ax = plt.subplots(1, 1)
attr = dfX[feat].values
ax.plot(attr, pMat[:,1], 'b*')
ax.plot(attr[indHBN, ], pMat[indHBN, 1], 'r*')
ax.plot([th, th], [np.nanmin(err), np.nanmax(err)], 'k-')
aa=pMat[:,1]
errL = aa[attr < th]
errR = aa[attr >= th]
ax.set_title('{}={:.2f} left={:.3f} right={:.3f}'.format(
    feat, th, np.nanmean(errL), np.nanmean(errR)))
fig.show()


# NITR_APP_KG_SQKM
# PHOS_APP_KG_SQKM
