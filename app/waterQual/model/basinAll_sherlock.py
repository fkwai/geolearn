from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.master import slurm
from hydroDL.data import gageII, usgs, gridMET

# wqData = waterQuality.DataModelWQ('basinAll')
# ind1 = wqData.indByRatio(0.8)
# ind2 = wqData.indByRatio(0.8, first=False)
# wqData.saveSubset(['first80', 'last20'], [ind1, ind2])

# devide to 8090 and 0010
wqData = waterQuality.DataModelWQ('basinAll')
indYr1 = waterQuality.indYr(
    wqData.info, yrLst=[1979, 2000])[0]
wqData.saveSubset('Y8090', indYr1)
indYr2 = waterQuality.indYr(
    wqData.info, yrLst=[2000, 2020])[0]
wqData.saveSubset('Y0010', indYr2)


caseLst = list()
subsetLst = ['Y8090', 'Y0010']
for subset in subsetLst:
    saveName = 'basinAll-{}-opt1'.format(subset)
    caseName = basins.wrapMaster(dataName='basinAll', trainName=subset, saveEpoch=50,
                                 batchSize=[None, 2000], outName=saveName)
    caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=48, nM=64)
