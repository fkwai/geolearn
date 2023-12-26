
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
yP1 = DF.extractSubset(yP, trainSet)
yP2 = DF.extractSubset(yP, testSet)
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
importlib.reload(utils.stat)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)

# box plot
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'svg.fonttype': 'none'})

rcParams =matplotlib.rcParams


# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

statStrLst=['Corr','KGE','MAPE','SMAPE','Bias','NSE','RMSE','MAE']
statStrLst=['Corr','KGE']

corrL2=utils.stat.calCorr(yP2,obs2)
indPlot = np.argsort(np.nanmean(matLR, axis=0))

for statStr in statStrLst:
    func=getattr(utils.stat,'cal'+statStr)
    statL1=func(yP1,obs1)
    statL2=func(yP2,obs2)
    statW1=func(yW1,obs1)
    statW2=func(yW2,obs2)
    for stat in [statL1, statL2, statW1, statW2]:
        stat[matRm] = np.nan
    codeStrLst = list()
    dataPlot = list()
    temp = list()
    for k in indPlot:
        code = codeLst[k]
        codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
        temp.append(code)
        dataPlot.append([statL2[:, k], statW2[:, k]])
    strLst = usgs.codeStrPlot(codeStrLst)
    cLst = ['#e41a1c', '#377eb8']

    fig, axes = figplot.boxPlot(
        dataPlot, widths=0.5, figsize=(12, 4), label1=strLst, cLst=cLst)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
    fig.show()

    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    plt.savefig(os.path.join(dirPaper, 'box_{}'.format(statStr.lower())))
    plt.savefig(os.path.join(dirPaper, 'box_{}.svg'.format(statStr.lower())))

# calculate corr between simplicity and error metrics
corrL2 = utils.stat.calCorr(yP2, obs2)
corrW2 = utils.stat.calCorr(yW2, obs2)
kgeL2 = utils.stat.calKGE(yP2, obs2)
kgeW2 = utils.stat.calKGE(yW2, obs2)


for corr in [ corrL2,  corrW2]:        
    corr[matRm] = np.nan
for kge in [kgeL2, kgeW2]:
    kge[matRm] = np.nan
    kge[np.isinf(kge)] = np.nan

a = np.nanmedian(matLR, axis=0)

np.corrcoef(a, np.nanmedian(corrL2, axis=0))
np.corrcoef(a, np.nanmedian(corrW2, axis=0))

np.corrcoef(a, np.nanmedian(kgeL2, axis=0))
np.corrcoef(a, np.nanmedian(kgeW2, axis=0))




