from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataName = 'N200'
label = 'QFPRT2C'
# DF = dbBasin.DataFrameBasin(dataName)
rho = 1000
nbatch = 500
hs = 256
trainSet = 'rmYr5'
testSet = 'pkYr5'
codeLst = usgs.varC

varX = dbBasin.label2var(label.split('2')[0])
mtdX = dbBasin.io.extractVarMtd(varX)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
# for code in codeLst:
for code in ['80154']:
    varY = [code]
    mtdY = dbBasin.io.extractVarMtd(varY)
    outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
    dictP = basinFull.wrapMaster(
        outName=outName, dataName=dataName, trainSet=trainSet,
        nEpoch=500, saveEpoch=50,
        varX=varX, varY=varY, varXC=varXC, varYC=varYC,
        mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC,
        hiddenSize=hs, batchSize=[rho, nbatch])
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    # slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
    basinFull.trainModel(outName)
