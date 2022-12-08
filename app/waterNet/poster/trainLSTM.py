from hydroDL import kPath
from hydroDL.data import dbBasin, gageII
from hydroDL.master import basinFull, slurm
import os

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = dbBasin.io.extractVarMtd(varX)
varY = ['runoff']
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = None

dataLst = ['QN90ref', 'Q95ref']
trainSet = 'WYB09'

rho = 365
for dataName in dataLst:
    outName = '{}-{}'.format(dataName, trainSet)
    dictP = basinFull.wrapMaster(outName=outName, dataName=dataName,
                                 trainSet=trainSet, optBatch='Random',
                                 varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                 nEpoch=1000, batchSize=[rho, 100],
                                 mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
