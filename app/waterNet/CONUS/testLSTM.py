

from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import numpy as np

dataName = 'QN90ref'
DF = dbBasin.DataFrameBasin(dataName)

trainSet = 'WYB09'
testSet = 'WYA09'
# LSTM
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=100)

Q = DF.extractSubset(DF.q, subsetName=testSet)
y = Q[:, :, 1]
k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(yL[:, k], 'b')
ax.plot(y[:, k], 'k')
fig.show()
