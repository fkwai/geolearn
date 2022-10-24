

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
dataName = 'camelsD'
trainSet = 'WY8095'
testSet = 'WY9510'
DF = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=500, batchSize=100)

Q = DF.extractSubset(DF.q, subsetName=testSet)
y = Q[:, :, 1]

nash2 = utils.stat.calNash(yL[:, :, 0], y)
corr2 = utils.stat.calCorr(yL[:, :, 0], y)

np.median(nash2)
np.median(corr2)
