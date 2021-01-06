import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn, transform


caseLst = ['050301',
           '080107',
           '080203',
           '080204',
           '080301',
           '080304',
           '080305',
           '080401',
           '080402',
           '080503',
           '090203',
           '090402']

dataName = 'Q90'
dm = dbBasin.DataModelFull(dataName)
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
nash0 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
rmse0 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
corr0 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
bias0 = utils.stat.calBias(yP[indT:, :, 0], yO[indT:, :, 0])

nashLst = list()
rmseLst = list()
corrLst = list()
biasLst = list()


for case in caseLst:
    testSet = 'EcoB'+case
    nashLstTemp = list()
    rmseLstTemp = list()
    corrLstTemp = list()
    biasLstTemp = list()
    trainLst = [case[:6], case[:4], case[:2]]
    outLst = ['{}-EcoB{}-B10-gs'.format(dataName, x)
              for x in trainLst]
    for outName in outLst:
        yP, ycP = basinFull.testModel(
            outName, DM=dm, batchSize=20, testSet=testSet)
        yO, ycO = basinFull.getObs(outName, testSet, DM=dm)
        nash2 = utils.stat.calNash(yP[indT:, :, 0], yO[indT:, :, 0])
        rmse2 = utils.stat.calRmse(yP[indT:, :, 0], yO[indT:, :, 0])
        corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])
        bias2 = utils.stat.calBias(yP[indT:, :, 0], yO[indT:, :, 0])
        nashLstTemp.append(nash2)
        rmseLstTemp.append(rmse2)
        corrLstTemp.append(corr2)
        biasLstTemp.append(bias2)
    # global model
    indS = [dm.siteNoLst.index(siteNo) for siteNo in dm.subset[testSet]]
    nashLstTemp.append(nash0[indS])
    rmseLstTemp.append(rmse0[indS])
    corrLstTemp.append(corr0[indS])
    biasLstTemp.append(bias0[indS])
    nashLst.append(nashLstTemp)
    rmseLst.append(rmseLstTemp)
    corrLst.append(corrLstTemp)
    biasLst.append(biasLstTemp)


# plot box
labLst = list()
for case in caseLst:
    labLst.append('{}.{}.{}'.format(
        int(case[:2]), int(case[2:4]), int(case[4:6])))
label1 = labLst
matLst = [rmseLst, corrLst, nashLst]
nameLst = ['rmse', 'corr', 'nash']
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'

# tempLst = ['080401', '080305', '080304', '090203', '080301', '050301']
tempLst = caseLst
rangeLst = [[0, 1], [0.7, 1], [0.4, 1]]
for kk in range(3):
    name = nameLst[kk]
    mat = [matLst[kk][caseLst.index(x)] for x in tempLst]
    yRange = rangeLst[kk]
    lab1 = [labLst[caseLst.index(x)] for x in tempLst]
    fig = figplot.boxPlot(mat, widths=0.5, cLst='ygbr',
                          label1=lab1, figsize=(12, 4), yRange=yRange)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    saveFile = os.path.join(saveFolder, 'q_sim_{}'.format(name))
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
    fig.show()


# significance test
columns = ['local vs near',
           'near vs far',
           'far vs dissimilar',
           'near vs dissimilar']
dfS = pd.DataFrame(index=labLst+['All'], columns=columns+['N'])
errLst = corrLst
aLst, bLst, cLst, dLst = [list() for k in range(4)]
for k, eco in enumerate(labLst):
    a = errLst[k][0]
    b = errLst[k][1]
    c = errLst[k][2]
    d = errLst[k][3]
    s, p = scipy.stats.wilcoxon(a, b)
    dfS.at[eco, 'local vs near'] = p
    s, p = scipy.stats.wilcoxon(b, c)
    dfS.at[eco, 'near vs far'] = p
    s, p = scipy.stats.wilcoxon(c, d)
    dfS.at[eco, 'far vs dissimilar'] = p
    s, p = scipy.stats.wilcoxon(b, d)
    dfS.at[eco, 'near vs dissimilar'] = p
    dfS.at[eco, 'N'] = len(a)
    aLst.append(a)
    bLst.append(b)
    cLst.append(c)
    dLst.append(d)
a = np.concatenate(aLst)
b = np.concatenate(bLst)
c = np.concatenate(cLst)
d = np.concatenate(dLst)
s, p = scipy.stats.wilcoxon(a, b)
dfS.at['All', 'local vs near'] = p
s, p = scipy.stats.wilcoxon(b, c)
dfS.at['All', 'near vs far'] = p
s, p = scipy.stats.wilcoxon(c, d)
dfS.at['All', 'far vs dissimilar'] = p
s, p = scipy.stats.wilcoxon(b, d)
dfS.at['All', 'near vs dissimilar'] = p
dfS.at['All', 'N'] = len(a)


# 05-03-01,212,176,135
# 08-01-07,1267,224,68
# 08-02-03,1267,190,93
# 08-02-04,1267,190,67
# 08-03-01,1267,402,72
# 08-03-04,1267,402,127
# 08-03-05,1267,402,97
# 08-04-01,1267,312,107
# 08-04-02,1267,312,61
# 08-05-03,1267,139,90
# 09-02-03,532,189,102
# 09-04-02,532,252,82
