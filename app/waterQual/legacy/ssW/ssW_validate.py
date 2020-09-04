from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)

code = '00945'
label = 'ntnS'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
statTup = basins.loadStat(outName)
master = basins.loadMaster(outName)

yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData)
obsLst1 = wqData.extractSubset(subset=trainSet)
obsLst2 = wqData.extractSubset(subset=testSet)
indC = wqData.varC.index(code)
ycT1 = obsLst1[3][:, indC:indC+1]
ycT2 = obsLst2[3][:, indC:indC+1]
yT1 = obsLst1[2][:, :, 0:1]
yT2 = obsLst2[2][:, :, 0:1]

errMatC1 = wqData.errBySiteC(
    ycP1, subset=trainSet, varC=master['varYC'])
errMatC2 = wqData.errBySiteC(
    ycP2, subset=testSet, varC=master['varYC'])
# errMatQ1 = wqData.errBySiteQ(
#     yP1, subset=trainSet, varQ=master['varY'])
# errMatQ2 = wqData.errBySiteQ(
#     yP2, subset=testSet, varQ=master['varY'])

# np.nanmean(errMatQ2[:, 0, 1])
np.nanmean(errMatC1[:, 0, 1])
np.nanmean(errMatC2[:, 0, 1])

# transfer - validate if training error is correct
mtd = wqData.extractVarMtd(master['varYC'])
xcP = transform.transInAll(ycP2, mtd, statLst=statTup[3])
xcT = transform.transInAll(ycT2, mtd, statLst=statTup[3])
mtd = wqData.extractVarMtd(master['varY'])
xP = transform.transInAll(yP2, mtd, statLst=statTup[2])
xT = transform.transInAll(yT2, mtd, statLst=statTup[2])

np.sqrt(np.nanmean((xT-xP)**2))
np.sqrt(np.nanmean((xcT-xcP)**2))
(np.sqrt(np.nanmean((xT-xP)**2))+np.sqrt(np.nanmean((xcT-xcP)**2)))/2

# see correlation
info = wqData.subsetInfo(testSet)
siteNoLst = info.siteNo.unique()
corrMat = np.full([len(siteNoLst), 2], np.nan)
for i, siteNo in enumerate(siteNoLst):
    indS = info[info['siteNo'] == siteNo].index.values
    a = xcT[indS, 0]
    b = xcP[indS, 0]
    _, indV = utils.rmNan([a, b])
    corrMat[i, 1] = np.corrcoef(a[indV], b[indV])[0, 1]
    a = xT[-1, indS, 0]
    b = xP[-1, indS, 0]
    _, indV = utils.rmNan([a, b])
    corrMat[i, 0] = np.corrcoef(a[indV], b[indV])[0, 1]
np.mean(corrMat[:, 1])
