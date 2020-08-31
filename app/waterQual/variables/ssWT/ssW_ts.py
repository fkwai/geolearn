from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataName = 'ssWT'
wqData = waterQuality.DataModelWQ(dataName)
caseLst = list()

varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
codeLst = varNtnUsgsLst

code = '00940'
ep = 500
label = 'ntnS'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
master = basins.loadMaster(outName)
yP1, ycP1 = basins.testModel(outName, trainSet, wqData=wqData, ep=ep)
yP2, ycP2 = basins.testModel(outName, testSet, wqData=wqData, ep=ep)
errMatC1 = wqData.errBySiteC(ycP1, subset=trainSet, varC=master['varYC'])
errMatC2 = wqData.errBySiteC(ycP2, subset=testSet, varC=master['varYC'])
obsLst1 = wqData.extractSubset(subset=trainSet)
obsLst2 = wqData.extractSubset(subset=testSet)
indC = wqData.varC.index(code)
ycT1 = obsLst1[3][:, indC]
ycT2 = obsLst2[3][:, indC]

info1 = wqData.subsetInfo(trainSet)
info2 = wqData.subsetInfo(testSet)
siteNoLst = info1['siteNo'].unique().tolist()

iP = 10
siteNo = siteNoLst[iP]
ind1 = info1.loc[info1['siteNo'] == siteNo].index.values
ind2 = info2.loc[info2['siteNo'] == siteNo].index.values
t1 = info1.iloc[ind1]['date'].values
t2 = info2.iloc[ind2]['date'].values
p1 = ycP1[ind1, 0]
p2 = ycP2[ind2, 0]
o1 = ycT1[ind1]
o2 = ycT2[ind2]
t1, p1, o1 = utils.rmNan([t1, p1, o1])
t2, p2, o2 = utils.rmNan([t2, p2, o2])

fig, axes = plt.subplots(2, 1)
styLst = ['--*', '--*']
axplot.plotTS(axes[0], t1, [o1, p1], cLst='kr', styLst=styLst)
axplot.plotTS(axes[1], t2, [o2, p2], cLst='kr', styLst=styLst)
fig.show()

# plot yr by yr
fig1 = figplot.tsYr(t1, [o1, p1], cLst='kr', showCorr=True)
fig1.show()
fig2 = figplot.tsYr(t2, [o2, p2], cLst='kr', showCorr=True)
fig2.show()

# _, indV1 = utils.rmNan([p1, o1])
# np.corrcoef(p1[indV1], o1[indV1])[0, 1]
# _, indV2 = utils.rmNan([p2, o2])
# np.corrcoef(p2[indV2], o2[indV2])[0, 1]
# iP0 = wqData.siteNoLst.index(siteNo)
# errMatC1[iP0, :, :]
# errMatC2[iP0, :, :]

# fig, axes = plt.subplots(1, 2)
# axes[0].plot(p1, o1, '*')
# axes[1].plot(p2, o2, '*')
# fig.show()
