
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
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)
importlib.reload(utils.stat)

mapeL1 = utils.stat.calMAPE(yP1, obs1)
mapeL2 = utils.stat.calMAPE(yP2, obs2)
mapeW1 = utils.stat.calMAPE(yW1, obs1)
mapeW2 = utils.stat.calMAPE(yW2, obs2)

smapeL1 = utils.stat.calSMAPE(yP1, obs1)
smapeL2 = utils.stat.calSMAPE(yP2, obs2)
smapeW1 = utils.stat.calSMAPE(yW1, obs1)
smapeW2 = utils.stat.calSMAPE(yW2, obs2)

maeL1 = np.nanmean(np.abs(yP1-obs1), axis=0)
maeL2 = np.nanmean(np.abs(yP2-obs2), axis=0)
maeW1 = np.nanmean(np.abs(yW1-obs1), axis=0)
maeW2 = np.nanmean(np.abs(yW2-obs2), axis=0)


rmseL1 = utils.stat.calRmse(yP1, obs1)
rmseL2 = utils.stat.calRmse(yP2, obs2)
rmseW1 = utils.stat.calRmse(yW1, obs1)
rmseW2 = utils.stat.calRmse(yW2, obs2)

biasL1 = np.nanmean(yP1, axis=0)-np.nanmean(obs1, axis=0)
biasL2 = np.nanmean(yP2, axis=0)-np.nanmean(obs2, axis=0)
biasW1 = np.nanmean(yW1, axis=0)-np.nanmean(obs1, axis=0)
biasW2 = np.nanmean(yW2, axis=0)-np.nanmean(obs2, axis=0)


# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan
for mape in [mapeL1, mapeL2, mapeW1, mapeW2]:
    mape[matRm] = np.nan
for bias in [biasL1, biasL2, biasW1, biasW2]:
    bias[matRm] = np.nan
for smape in [smapeL1, smapeL2, smapeW1, smapeW2]:
    smape[matRm] = np.nan
# box plot
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})

# re-order
indPlot = np.argsort(np.nanmean(corrL2, axis=0))
codeStrLst = list()
dataPlot = list()
temp = list()
for k in indPlot:
    code = codeLst[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    temp.append(code)
    dataPlot.append([corrL2[:, k], corrW2[:, k]])

importlib.reload(usgs)
strLst = usgs.codeStrPlot(codeStrLst)
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=strLst)
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()

# re-order
indPlot = np.argsort(np.nanmedian(mapeL2, axis=0))
codeStrLst = list()
dataPlot = list()
temp = list()
for k in indPlot:
    code = codeLst[k]
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
    temp.append(code)
    dataPlot.append([mapeL2[:, k]*100, mapeW2[:, k]*100])
importlib.reload(usgs)
strLst = usgs.codeStrPlot(codeStrLst)
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), label1=strLst)
plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
fig.show()
figFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
fig.savefig(os.path.join(figFolder, 'box_smape'.format(label, trainSet)))
fig.savefig(os.path.join(figFolder, 'box_smape.svg'.format(label, trainSet)))


# re-order
codeGroup = [
    ['00405', '00400', '71846', '00660', '00605',
     '00665', '80154', '00600', '00618', '00681'],
    ['00935', '00955', '00945', '00940', '00095',
     '00925', '00915', '00930', '00300', '00010']]
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})
for rmse in [rmseL1, rmseL2, rmseW1, rmseW2]:
    rmse[matRm] = np.nan
indPlot = np.argsort(np.nanmean(biasL2, axis=0))
for codeG in codeGroup:
    codeStrLst = list()
    dataPlot = list()
    for code in codeG:
        codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
        k = DF.varC.index(code)
        dataPlot.append([biasL2[:, k], biasW2[:, k]])
    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    importlib.reload(usgs)
    strLst = usgs.codeStrPlot(codeStrLst)
    fig, axes = figplot.boxPlot(
        dataPlot, widths=0.5, figsize=(12, 4), label1=strLst, sharey=False)
    for ax in axes:
        ax.axhline(0, color='k')
    plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
    fig.tight_layout()
    fig.show()


x = DF.g[:, 0]
fig, ax = plt.subplots(1, 1)
ax.hist(x, normed=True,bins=100, cumulative=True, label='CDF',
        histtype='step', alpha=0.8, color='k')
ax.set_xlabel('area [sqkm]')
ax.set_ylabel('# basins')
fig.show()

len(np.where(x<2000)[0])