from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('HBN')


outLst = ['HBN-first50-opt1', 'HBN-first50-opt2']
trainSet = 'first50'
testSet = 'last50'
pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
for outName in outLst:
    p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
    p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
    errMat1 = wqData.errBySite(p1, subset=trainSet)
    errMat2 = wqData.errBySite(p2, subset=testSet)
    pLst1.append(p1)
    pLst2.append(p2)
    errMatLst1.append(errMat1)
    errMatLst2.append(errMat2)

# plot
codeSel = ['00955', '00940', '00915']
# codeSel = ['00600', '00605', '00405']
siteNoLst = wqData.info['siteNo'].unique().tolist()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = usgs.codePdf


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        shortName = codePdf.loc[code]['shortName']
        title = 'correlation of {} {}'.format(shortName, code)
        axplot.mapPoint(axM[k], lat, lon,
                        errMat2[:, ic, 1], s=12, title=title)
    figP, axP = plt.subplots(len(codeSel), 3, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    # cp = np.concatenate([p1[ind1], p2[ind2]])
    # ct = np.concatenate([o1[ind1], o2[ind2]])
    # q = wqData.q[-1, np.concatenate([ind1, ind2]), 0]
    cLst = [o2[ind2]]+[p[ind2] for p in pLst2]
    q = wqData.q[-1, ind2, 0]

    x = 10**np.linspace(np.log10(np.min(q[q > 0])),
                        np.log10(np.max(q[~np.isnan(q)])), 20)
    for k, code in enumerate(codeSel):
        ic = wqData.varC.index(code)
        for j, c, s in zip([0, 1, 2], cLst, ['obs', 'op1', 'opt2']):
            sa, sb, ys = relaCQ.slopeModel(q, c[:, ic], x)
            ceq, dw, yk = relaCQ.kateModel(q, c[:, ic], x)
            title = '{} [{:.2f},{:.2f}], [{:.2f},{:.2f}]'.format(
                s, sa, sb, ceq, dw)
            axP[k, j].plot(np.log10(q), c[:, ic], '*k',  label='obs')
            axP[k, j].plot(np.log10(x), ys, '-b', label='slope')
            axP[k, j].plot(np.log10(x), yk, '-r', label='kate')
            axP[k, j].set_title(title)
            # axP[k, j].set_xticks([])


figplot.clickMap(funcMap, funcPoint)
