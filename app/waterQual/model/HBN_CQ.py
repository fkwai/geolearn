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

figFolder = os.path.join(kPath.dirWQ, 'HBN', 'CQ')

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

siteNoLst = wqData.info['siteNo'].unique().tolist()
codePdf = usgs.codePdf
pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
unitConv = 0.3048**3*365*24*60*60/1000**2


ns = len(siteNoLst)
nc = len(wqData.varC)
matSa = np.full([ns, nc, 3], np.nan)
matSb = np.full([ns, nc, 3], np.nan)
matCeq = np.full([ns, nc, 3], np.nan)
matDw = np.full([ns, nc, 3], np.nan)

for iS, siteNo in enumerate(siteNoLst):
    info1 = wqData.subsetInfo(trainSet)
    info2 = wqData.subsetInfo(testSet)
    ind1 = info1[info1['siteNo'] == siteNo].index
    ind2 = info2[info2['siteNo'] == siteNo].index
    cLst = [o2[ind2]]+[p[ind2] for p in pLst2]
    area = pdfArea.loc[siteNo]['DRAIN_SQKM']
    q = wqData.q[-1, ind2, 0]/area*unitConv

    for code in wqData.varC:
        iC = wqData.varC.index(code)
        for k, c in enumerate(cLst):
            try:
                sa, sb, ys = relaCQ.slopeModel(q, c[:, iC])
                ceq, dw, yk = relaCQ.kateModel(q, c[:, iC])
                matSa[iS, iC, k] = sa
                matSb[iS, iC, k] = sb
                matCeq[iS, iC, k] = ceq
                matDw[iS, iC, k] = dw
            except:
                pass

#
for code in wqData.varC:
    iC = wqData.varC.index(code)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(matSa[:, iC, 0], matSa[:, iC, 1], 'r*')
    axes[0, 0].plot(matSa[:, iC, 0], matSa[:, iC, 2], 'b*')
    axes[0, 0].set_title('a')
    axes[0, 0].axis('equal')

    axes[0, 1].plot(matSb[:, iC, 0], matSb[:, iC, 1], 'r*')
    axes[0, 1].plot(matSb[:, iC, 0], matSb[:, iC, 2], 'b*')
    axes[0, 1].set_title('b')
    axes[0, 1].axis('equal')

    axes[1, 0].plot(matCeq[:, iC, 0], matCeq[:, iC, 1], 'r*')
    axes[1, 0].plot(matCeq[:, iC, 0], matCeq[:, iC, 2], 'b*')
    axes[1, 0].set_title('ceq')
    axes[1, 0].axis('equal')

    axes[1, 1].plot(matDw[:, iC, 0], matDw[:, iC, 1], 'r*')
    axes[1, 1].plot(matDw[:, iC, 0], matDw[:, iC, 2], 'b*')
    axes[1, 1].set_title('dw')
    axes[1, 1].axis('equal')

    shortName = codePdf.loc[code]['shortName']
    fig.suptitle('{} {}'.format(shortName, code))

    fig.show()
    figName = code
    fig.savefig(os.path.join(figFolder, figName))
