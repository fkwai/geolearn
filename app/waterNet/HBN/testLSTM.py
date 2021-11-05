
from hydroDL.data import dbBasin, gageII, gridMET
from hydroDL.master import basinFull
import numpy as np
from hydroDL import utils
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dataName = 'HBN_Q90ref'
DF = dbBasin.DataFrameBasin(dataName)
label = 'test'
varX = gridMET.varLst
# varX = ['pr', 'etr', 'tmmn', 'tmmx']
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['runoff']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'WYB09'
testSet = 'WYA09'
rho = 365
outName = '{}-{}'.format(dataName, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=1000, batchSize=[rho, 100], nIterEp=5,
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
basinFull.trainModel(outName)
#

yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', reTest=True)
yP = yP[:, :, 0]
yO = DF.extractT(['runoff'])
yO = yO[:, :, 0]
indT = np.where(DF.t == np.datetime64('2010-01-01'))[0][0]

k = 1
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, DF.t, [yO[:, k], yP[:, k]])
fig.show()

fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, DF.t[indT:], [yO[indT:, k], yP[indT:, k]])
fig.show()

nash = utils.stat.calNash(yP[indT:, :], yO[indT:, :])
corr = utils.stat.calCorr(yP[indT:, :], yO[indT:, :])
np.mean(nash)
np.median(nash)
np.mean(corr)
np.median(corr)
