
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, transform, dbBasin
import os
import json

dataName = 'Q90ref'
globalName = '{}-B10'.format(dataName)

modelFolder = basinFull.nameFolder(outName)
masterFile = os.path.join(modelFolder, 'master.json')
with open(masterFile, 'r') as fp:
    master = json.load(fp)

defaultMaster = dict(
    dataName='test', trainName='all', outName=None,
    hiddenSize=256, batchSize=[365, 500],
    nEpoch=500, saveEpoch=100, resumeEpoch=0,
    optNaN=[1, 1, 0, 0], overwrite=True,
    modelName='CudnnLSTM', crit='RmseLoss', optim='AdaDelta',
    varX=gridMET.varLst, varXC=gageII.lstWaterQuality,
    varY=['00060'], varYC=None,
    sd='1979-01-01', ed='2010-01-01', subset='all', borrowStat=None
)

caseName = basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                                varY=varY, varXC=varXC, varYC=varYC, sd=sd, ed=ed,
                                subset=subset, borrowStat=globalName)

mm = defaultMaster.copy()
out = mm.update(master)
