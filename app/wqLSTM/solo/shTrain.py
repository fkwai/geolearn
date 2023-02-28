from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

labelLst = ['FT2QC', 'QFT2C', 'QT2C']
# trainLst = ['B15']+['rmYr5b{}'.format(k) for k in range(5)]+['rmRT5b{}'.format(k) for k in range(5)]
# testLst = ['A15']+['pkYr5b{}'.format(k) for k in range(5)]+['pkRT5b{}'.format(k) for k in range(5)]

trainLst = ['rmYr5b0']
testLst = ['pkYr5b0']
codeLst = ['00915', '00955', '00618']


def label2var(label, code):
    varF = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
    varT = ['datenum', 'sinT', 'cosT']
    varQ = ['runoff']
    if label == 'FT2QC':
        varX = varF + varT
        varY = varQ + [code]
    elif label == 'QFT2C':
        varX = varQ + varF + varT
        varY = [code]
    elif label == 'QT2C':
        varX = varQ + varT
        varY = [code]
    varXC = gageII.varLst
    varYC = None
    return varX, varY, varXC, varYC


for code in ['00915', '00955', '00618']:
    dataName = '{}-{}'.format(code, 'B200')
    for label in labelLst:
        for trainSet in trainLst:
            varX, varY, varXC, varYC = label2var(label, code)
            mtdX = dbBasin.io.extractVarMtd(varX)
            mtdY = dbBasin.io.extractVarMtd(varY)
            mtdXC = dbBasin.io.extractVarMtd(varXC)
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
                saveEpoch=20,
            )
            cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
            slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
            # basinFull.trainModel(outName)
