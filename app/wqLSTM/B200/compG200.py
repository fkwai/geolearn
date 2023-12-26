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

trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'

# new LSTM model
dataName='rmTK-B200'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
ep1=400
label='QFT2C'
DF1 = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}-{}'.format('rmTK-B200', label, trainSet)
dictMaster = basinFull.loadMaster(outName)
yN1, ycN1 = basinFull.testModel(outName, DF=DF1, testSet=testSet, ep=ep1)
matObs1 = DF1.c
obs1 = DF1.extractSubset(matObs1, testSet)
# count
matB = (~np.isnan(DF1.c)*~np.isnan(DF1.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF1.extractSubset(matB, trainSet)
matB2 = DF1.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm1 = (count1 < 80) | (count2 < 20)

# old LSTM model
label='QFPRT2C'
trainSet= 'rmYr5'
testSet = 'pkYr5'
dataName='G200'
ep2=1000
DF2 = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}-{}'.format(dataName, label, trainSet)
dictMaster = basinFull.loadMaster(outName)
yO2, ycO2 = basinFull.testModel(outName, DF=DF2, testSet=testSet, ep=ep2)
matObs2 = DF2.c
obs2 = DF2.extractSubset(matObs2, testSet)
# count
matB = (~np.isnan(DF2.c)*~np.isnan(DF2.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF2.extractSubset(matB, trainSet)
matB2 = DF2.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm2 = (count1 < 80) | (count2 < 20)


# error metrics
importlib.reload(utils.stat)
# strFunc='calLogMAE'
strFunc='calCorr'
errFunc=getattr(utils.stat,strFunc)

stat1=errFunc(yN1, obs1)
stat1[matRm1]=np.nan
stat2=errFunc(yO2, obs2)
stat2[matRm2]=np.nan

# plot
dataPlot = list()
codeStrLst = list()
for ic,code in enumerate(usgs.varC):
    dataPlot.append([stat1[:,ic], stat2[:,ic]])
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), 
    label1=codeStrLst,label2=['new','old'])
fig.show()

# with same sites
siteNoLst1 = DF1.siteNoLst
siteNoLst2 = DF2.siteNoLst
# find index of common sites
siteNoLst=list(set(siteNoLst1).intersection(siteNoLst2))
ind1 = [siteNoLst1.index(siteNo) for siteNo in siteNoLst]
ind2 = [siteNoLst2.index(siteNo) for siteNo in siteNoLst]
dataPlot = list()
codeStrLst = list()
for ic,code in enumerate(usgs.varC):
    dataPlot.append([stat1[ind1,ic], stat2[ind2,ic]])
    codeStrLst.append(usgs.codePdf.loc[code]['shortName'])
fig, axes = figplot.boxPlot(
    dataPlot, widths=0.5, figsize=(12, 4), 
    label1=codeStrLst,label2=['new','old'])
fig.show()
