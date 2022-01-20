from hydroDL import kPath
from hydroDL.data import gageII, gridMET, dbBasin
from hydroDL.master import slurm
from hydroDL.master import basinFull


dataName = 'B5Y09-00955'
outName = 'B5Y09-00955-C'
varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI', 'runoff']
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['00955']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

trainSet = 'WYB09'
testSet = 'WYA09'
rho = 1000
# outName = '{}-{}'.format(dataName, trainSet)
dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                             trainSet=trainSet,
                             varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                             nEpoch=1000, batchSize=[rho, 100],
                             mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
# basinFull.trainModel(outName)
cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
