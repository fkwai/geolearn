from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
import numpy as np
from hydroDL.master import slurm
import importlib


# ntn variables
dataName = 'rbWN5'
# wqData = waterQuality.DataModelWQ(dataName)

varF = gridMET.varLst
varQ = ['runoff']
varP = ntn.varLst
varT = ['sinT', 'cosT']

code = 'comb'
label = 'QFP_C'
varX = varQ+varF+varP
varY = None
varYC = usgs.newC
dictRegion = {
    'PNV': [2, 3, 4, 5, 9, 11],
    'NUTR': [2, 3, 4, 5, 6, 7, 8, 9, 11, 14],
    'HLR': [3, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 20],
    'ECO': [5.3, 6.2, 8.1, 8.2, 8.3, 8.4, 9.2, 9.3, 9.4, 10.1, 11.1]
}
caseLst = list()
for region in list(dictRegion.keys()):
    for regionId in dictRegion[region]:
        if region == 'ECO':
            idLst = [int(x) for x in str(regionId).split('.')]
            regionId = '{:02d}{:02d}'.format(*idLst)
        else:
            regionId = '{:02d}'.format(regionId)

        trainSet = 'comb-{}{}-B10'.format(region, regionId)
        saveName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        caseName = basins.wrapMaster(
            dataName=dataName, trainName=trainSet, batchSize=[None, 500],
            outName=saveName, varX=varX, varY=varY, varYC=varYC)
        caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=8)

# basins.trainModelTS(caseName)
