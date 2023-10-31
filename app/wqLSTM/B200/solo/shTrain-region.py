from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS
from hydroDL.master import slurm
from hydroDL.data import dbBasin
from hydroDL.master import basinFull


trainLst = ['rmYr5b0_HUC{:02d}'.format(h) for h in range(1,18)]
testLst = ['pkYr5b0_HUC{:02d}'.format(h) for h in range(1,18)]
label='QFT2C'

codeLst = usgs.varC.copy()
codeExist=['00915', '00618']
for code in codeExist:
    codeLst.remove(code)

def label2var(label, code): # for a code
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


for code in codeLst:
    dataName = '{}-{}'.format(code, 'B200')
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
            saveEpoch=100,
        )
        cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
        slurm.submitJobGPU(outName, cmdP.format(outName), nH=8, nM=64)
            # basinFull.trainModel(outName)
