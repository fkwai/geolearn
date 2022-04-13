from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

dataNameLst = ['Y28N5', 'Y28N5rmTK']
# labelLst = ['FPRT2QC', 'QFPRT2C', 'QT2C']
labelLst = ['F2C', 'FQ2C']
trainSet = 'B10'
testSet = 'A10'
# DF = dbBasin.DataFrameBasin(dataName)

for dataName in dataNameLst:
    for label in labelLst:
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
#       basinFull.trainModel(outName)
#
