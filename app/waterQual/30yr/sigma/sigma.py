from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, gageII
import numpy as np
from hydroDL.master import slurm
import importlib
from hydroDL.model import rnn, crit, trainTS

dataName = 'test'
varQ = ['00060']
varP = ntn.varLst
varT = ['sinT', 'cosT']
varF = gridMET.varLst


varX = varQ+varF+varP
varXC = gageII.varLst
varY = varQ
varYC = usgs.newC

saveName = 'test'
trainSet = 'comb-A10'
outName = basins.wrapMaster(
    dataName=dataName, trainName=trainSet, batchSize=[None, 500],
    outName=saveName, varX=varX, varY=varY, varYC=varYC,
    crit='SigmaLoss',
    nEpoch=10, saveEpoch=10)

wqData = waterQuality.DataModelWQ('test')
basins.trainModelTS(outName)

importlib.reload(basins)
yp, sp, ycp, scp = basins.testModel(
    outName, trainSet, wqData=wqData, ep=10,reTest=True)
