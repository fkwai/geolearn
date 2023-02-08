from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataName = 'NY5'
labelLst = ['FT2QC', 'QFT2C', 'QT2C']
# trainLst = ['B15']+['rmYr5b{}'.format(k) for k in range(5)]+['rmRT5b{}'.format(k) for k in range(5)]
# testLst = ['A15']+['pkYr5b{}'.format(k) for k in range(5)]+['pkRT5b{}'.format(k) for k in range(5)]

trainLst = ['B15', 'rmYr5b0', 'rmRT5b0']
testLst = ['A15', 'pkYr5b0', 'pkRT5b0']

# DF = dbBasin.DataFrameBasin(dataName)


for label in labelLst:
    for trainSet in trainLst:
        varX = dbBasin.label2var(label.split('2')[0])
        mtdX = dbBasin.io.extractVarMtd(varX)
        varY = dbBasin.label2var(label.split('2')[1])
        mtdY = dbBasin.io.extractVarMtd(varY)
        varXC = gageII.varLst
        mtdXC = dbBasin.io.extractVarMtd(varXC)
        varYC = None
        mtdYC = dbBasin.io.extractVarMtd(varYC)
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        dictP = basinFull.wrapMaster(
            outName=outName,
            dataName=dataName,
            trainSet=trainSet,
            nEpoch=500,
            batchSize=[1000, 100],
            varX=varX,
            varY=varY,
            varXC=varXC,
            varYC=varYC,
            mtdX=mtdX,
            mtdY=mtdY,
            mtdXC=mtdXC,
            mtdYC=mtdYC,
            saveEpoch=20
        )
        cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
        slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
        # basinFull.trainModel(outName)
