from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataName = 'G200'
labelLst = ['QFPRT2C']
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']
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
        dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                     nEpoch=2000, batchSize=[365, 500], nIterEp=50,
                                     varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                     mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
        cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
        slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
        # basinFull.trainModel(outName)
