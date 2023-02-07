from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataName = 'NY5'
trainSet = 'B15'
varX = gridMET.varLst
mtdX = dbBasin.io.extractVarMtd(varX)
varY = usgs.varC
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
outName = 'test'
dictP = basinFull.wrapMaster(
    outName=outName,
    dataName=dataName,
    trainSet=trainSet,
    nEpoch=500,
    batchSize=[365, 100],
    varX=varX,
    varY=varY,
    varXC=varXC,
    varYC=varYC,
    mtdX=mtdX,
    mtdY=mtdY,
    mtdXC=mtdXC,
    mtdYC=mtdYC,
    saveEpoch=20,
    nIterEp=2,
)
cmdP = (
    'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
)
# slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
basinFull.trainModel(outName)
