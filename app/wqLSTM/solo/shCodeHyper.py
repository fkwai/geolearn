from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull

labelLst = ['FT2QC', 'QFT2C', 'QT2C']
# trainLst = ['B15']+['rmYr5b{}'.format(k) for k in range(5)]+['rmRT5b{}'.format(k) for k in range(5)]
# testLst = ['A15']+['pkYr5b{}'.format(k) for k in range(5)]+['pkRT5b{}'.format(k) for k in range(5)]

trainLst = ['rmYr5b0']
testLst = ['pkYr5b0']


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


def trainModel(code, dr, hs, rho, nLayer):
    dataName = '{}-{}'.format(code, 'B200')
    varX, varY, varXC, varYC = label2var(label, code)
    mtdX = dbBasin.io.extractVarMtd(varX)
    mtdY = dbBasin.io.extractVarMtd(varY)
    mtdXC = dbBasin.io.extractVarMtd(varXC)
    mtdYC = dbBasin.io.extractVarMtd(varYC)
    outName = '{}-{}-{}-d{}-h{}-rho{}-nl{}'.format(
        dataName, label, trainSet, dr, hs, rho, nLayer
    )
    dictP = basinFull.wrapMaster(
        outName=outName,
        dataName=dataName,
        trainSet=trainSet,
        nEpoch=500,
        batchSize=[rho, 100],
        varX=varX,
        varY=varY,
        varXC=varXC,
        varYC=varYC,
        mtdX=mtdX,
        mtdY=mtdY,
        mtdXC=mtdXC,
        mtdYC=mtdYC,
        saveEpoch=20,
        dropout=dr,
        hiddenSize=hs,
        nLayer=nLayer,
    )
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
    # basinFull.trainModel(outName)


drLst = [0.25, 0.75]
hsLst = [16, 64, 128]
rhoLst = [182, 365, 2000]
nLayerLst = [1, 2, 3]

# codeLst = ['00618','00915','00955']

code = '00915'
for label in labelLst:
    for trainSet in trainLst:
        for dr in drLst:
            for hs in hsLst:
                for rho in rhoLst:
                    for nLayer in nLayerLst:
                        trainModel(code, dr, hs, rho, nLayer)        
