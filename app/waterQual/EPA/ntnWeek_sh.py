from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm

dataName = 'refWeek'
wqData = waterQuality.DataModelWQ(dataName)
indYrO, indYrE = waterQuality.indYrOddEven(wqData.info)
wqData.saveSubset('Yodd', indYrO)
wqData.saveSubset('Yeven', indYrE)

codeLst = usgs.varC
subsetLst = ['Yodd', 'Yeven']
varX1 = gridMET.varLst
varX2 = gridMET.varLst+ntn.varLst+['distNTN']
nEp = 500
caseLst = list()
for subset in subsetLst:
    saveName = '{}-{}'.format(dataName, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 200], nEpoch=nEp,
        outName=saveName, varX=varX1)
    caseLst.append(caseName)
    saveName = '{}-{}-ntn'.format(dataName, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 200], nEpoch=nEp,
        outName=saveName,  varX=varX2)
    caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)
