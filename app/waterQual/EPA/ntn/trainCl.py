from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
import numpy as np

dataName = 'chloride'
wqData = waterQuality.DataModelWQ(dataName)
indYrO, indYrE = waterQuality.indYrOddEven(wqData.info)
wqData.saveSubset('Yodd', indYrO)
wqData.saveSubset('Yeven', indYrE)
codeLst = ['00940']
# subsetLst = ['Yodd', 'Yeven']
subsetLst = ['Yodd']
varXC = ['DRAIN_SQKM', 'SNOW_PCT_PRECIP', 'STREAMS_KM_SQ_KM', 'PCT_1ST_ORDER',
         'BFI_AVE', 'CONTACT', 'FORESTNLCD06', 'HLR_BAS_DOM_100M', 'ELEV_MEAN_M_BASIN',
         'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE', 'SLOPE_PCT']
varX1 = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr',
         'ph', 'Conduc', 'Cl']
varX2 = ['pr', 'sph', 'srad', 'tmmn', 'tmmx', 'pet', 'etr']
nEp = 100
caseLst = list()
for subset in subsetLst:
    saveName = '{}-{}-ntn'.format(dataName, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 100], nEpoch=nEp,
        outName=saveName, varXC=varXC, varX=varX1, varYC=codeLst,
        optNaN=[2, 2, 0, 0])
    caseLst.append(caseName)
    saveName = '{}-{}'.format(dataName, subset)
    caseName = basins.wrapMaster(
        dataName=dataName, trainName=subset, batchSize=[None, 100], nEpoch=nEp,
        outName=saveName, varXC=varXC, varX=varX2, varYC=codeLst, optNaN=[2, 2, 0, 0])
    caseLst.append(caseName)

for caseName in caseLst:
    basins.trainModelTS(caseName)
