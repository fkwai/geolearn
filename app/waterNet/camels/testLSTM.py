

import torch
import os
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
from hydroDL.model import trainBasin
import numpy as np
from hydroDL import utils

# LSTM
dataLst = ['camelsN', 'camelsD', 'camelsM']
labLst = ['NLDAS', 'dayMet', 'Maurer']
trainSet = 'WY8095'
testSet = 'WY9510'
yLst = list()
for dataName in dataLst:
    DF = dbBasin.DataFrameBasin(dataName)
    outName = '{}-{}'.format(dataName, trainSet)
    yL, ycL = basinFull.testModel(
        outName, DF=DF, testSet=testSet, reTest=True, ep=500, batchSize=100)
    yLst.append(yL)

Q = DF.extractSubset(DF.q, subsetName=testSet)
y = Q[:, :, 1]

nashLst = list()
corrLst = list()
for yL in yLst:
    nash = utils.stat.calNash(yL[:, :, 0], y)
    corr = utils.stat.calCorr(yL[:, :, 0], y)
    nashLst.append(nashLst)
    corrLst.append(corrLst)


fig, axes = figplot.boxPlot(
    [nashLst, corrLst], label1=['nash', 'corr'], 
    label2=['NLDAS', 'dayMet', 'Maurer'])
fig.show()