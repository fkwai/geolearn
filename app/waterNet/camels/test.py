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
dataLst = ['camelsN', 'camelsD', 'camelsM']
rho = 365
# for dataName in dataLst:
dataName = 'camelsN'
outName = '{}-{}'.format(dataName, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             trainSet=trainSet,nIterEp=20,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=1000, batchSize=[rho, 100],
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
basinFull.trainModel(outName)
# cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
# slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
