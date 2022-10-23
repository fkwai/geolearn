import matplotlib.pyplot as plt
from hydroDL import kPath
from hydroDL.data import dbBasin, camels
from hydroDL.master import basinFull, slurm


varX = camels.varF
mtdX = camels.extractVarMtd(varX)
varY = ['runoff']
mtdY = camels.extractVarMtd(varY)
varXC = camels.varG
mtdXC = camels.extractVarMtd(varXC)
varYC = None
mtdYC = None

trainSet = 'B05'
testSet = 'A05'

rho = 365
dataName = 'camelsTemp'
outName = '{}-{}'.format(dataName, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=100, batchSize=[rho, 100],
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
basinFull.trainModel(outName)

DF = dbBasin.DataFrameBasin(dataName)
outName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    outName, DF=DF, testSet=testSet, reTest=True, ep=100)

Q = DF.extractSubset(DF.q, subsetName=testSet)
y = Q[:, :, 1]
k = 0
fig, ax = plt.subplots(1, 1)
ax.plot(yL[:, k, 0], 'b')
ax.plot(y[:, k], 'k')
fig.show()
