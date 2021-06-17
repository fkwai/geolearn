from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataNameLst = ['G200Norm', 'G400Norm']

for dataName in dataNameLst:
    DF = dbBasin.DataFrameBasin(dataName)
    varX = DF.varF+DF.varQ
    mtdX = dbBasin.io.extractVarMtd(varX)
    varY = [c+'-N' for c in usgs.newC]
    mtdY = dbBasin.io.extractVarMtd(varY)
    varXC = DF.varG
    mtdXC = dbBasin.io.extractVarMtd(varXC)
    varYC = None
    mtdYC = dbBasin.io.extractVarMtd(varYC)
    sd = '1982-01-01'
    ed = '2009-12-31'
    outName = dataName
    trainSet = 'rmRT20'
    testSet = 'pkRT20'
    dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                 varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                 nEpoch=500, batchSize=[365, 1000],
                                 mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=32)
