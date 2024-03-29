from hydroDL import kPath
from hydroDL.data import gageII, gridMET, dbBasin
from hydroDL.master import slurm
from hydroDL.master import basinFull


varYLst = [['runoff'], ['runoff', '00955']]
outLst = ['B5Y09-00955-Q', 'B5Y09-00955-QC']
dataName = 'B5Y09-00955'
for varY, outName in zip(varYLst, outLst):
    varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
    mtdX = dbBasin.io.extractVarMtd(varX)
    # varY = ['runoff']
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
