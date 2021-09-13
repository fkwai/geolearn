import os
from hydroDL.master import slurm
from hydroDL.master import basinFull
from hydroDL.data import gageII, dbBasin

dataName = 'G200'
# DF = dbBasin.DataFrameBasin(dataName)

# count for code
codeLst = ['00600', '00618', '00915', '00945', '00955']
pLst = [100, 75, 50, 25]
nyLst = [6, 8, 10]

outLst = list()
for code in codeLst:
    for ny in nyLst:
        for p in pLst:
            label = 'QFPRT2C'
            trainSet = '{}-n{}-p{}-B10'.format(code, ny, p)
            varX = dbBasin.label2var(label.split('2')[0])
            mtdX = dbBasin.io.extractVarMtd(varX)
            varY = [code]
            mtdY = dbBasin.io.extractVarMtd([code])
            varXC = gageII.varLst
            mtdXC = dbBasin.io.extractVarMtd(varXC)
            varYC = None
            mtdYC = [code]
            outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
            outFolder = basinFull.nameFolder(outName)
            modelFile = os.path.join(outFolder, 'modelState_ep100')
            dictP = basinFull.wrapMaster(outName=outName, dataName=dataName, trainSet=trainSet,
                                         nEpoch=500, batchSize=[365, 20], nIterEp=50,
                                         varX=varX, varY=varY, varXC=varXC, varYC=varYC,
                                         mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC)
            outFolder = basinFull.nameFolder(outName)
            modelFile = os.path.join(outFolder, 'modelState_ep100')
            if not os.path.exists(modelFile):
                outLst.append(outName)
print(outLst)
for outName in outLst:
    cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
    # slurm.submitJobGPU(outName, cmdP.format(outName))
    # basinFull.trainModel(outName)
