import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')
figFolder = os.path.join(kPath.dirWQ, 'basinRef')

# # compare
# nameLst = ['areaLT10', 'areaGT2500', 'eco0503', 'eco0902', 'nutr06', 'nutr08']
# for name in nameLst:
outName = 'basinRef-first50-opt2'
trainSet = 'first50'
testSet = 'last50'
errMatLst1, errMatLst2 = [list(), list()]
p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
errMat1 = wqData.errBySite(p1, subset=trainSet)
errMat2 = wqData.errBySite(p2, subset=testSet)
master = basins.loadMaster(outName)
varAll = master['varYC']


siteNoLst = wqData.info.siteNo.unique().tolist()
varG = ['DRAIN_SQKM', 'ECO2_BAS_DOM', 'NUTR_BAS_DOM',
        'HLR_BAS_DOM_100M']
tabG = gageII.readData(varLst=varG, siteNoLst=siteNoLst)

# codeLst = ['00955', '00915', '00940', '00618']
codeLst = ['00915', '00618']
name = 'HLR_BAS_DOM_100M'
fig, axes = plt.subplots(len(codeLst), 1)
codePdf = usgs.codePdf
cLst = 'rgbc'
for k, code in enumerate(codeLst):
    iC = master['varYC'].index(code)
    vLst = np.sort(tabG[name].unique()).tolist()
    dataBox = list()
    for v in vLst:
        siteNo = tabG[tabG[name] == v].index.tolist()
        ind = [siteNoLst.index(s) for s in siteNo]
        err = errMat2[ind, iC, 1]
        dataBox.append(err)
    ax = axplot.plotBox(axes[k], dataBox, labLst=vLst, c=cLst[k])
    ax.set_title(codePdf.loc[code]['shortName'])
fig.show()

# area
areaLst = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 30000]
area = tabG['DRAIN_SQKM']

plt.hist(area, bins=20)
plt.show()

codeLst = ['00915', '00618']
fig, axes = plt.subplots(len(codeLst), 1)
codePdf = usgs.codePdf
cLst = 'rgbc'
for k, code in enumerate(codeLst):
    iC = master['varYC'].index(code)
    vLst = np.sort(tabG[name].unique()).tolist()
    dataBox = list()
    for v1, v2 in zip(areaLst[:-1], areaLst[1:]):
        siteNo = tabG[(tabG['DRAIN_SQKM'] >= v1) & (
            tabG['DRAIN_SQKM'] < v2)].index.tolist()
        ind = [siteNoLst.index(s) for s in siteNo]
        err = errMat2[ind, iC, 1]
        dataBox.append(err)
        ax = axplot.plotBox(axes[k], dataBox, labLst=areaLst[1:], c=cLst[k])
        ax.set_title(codePdf.loc[code]['shortName'])
fig.show()

