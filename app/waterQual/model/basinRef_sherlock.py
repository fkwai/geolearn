from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL.master import slurm
from hydroDL.data import gageII, usgs, gridMET


wqData = waterQuality.DataModelWQ('basinRef')
ind1 = wqData.indByRatio(0.5)
ind2 = wqData.indByRatio(0.5, first=False)
wqData.saveSubset(['first50', 'last50'], [ind1, ind2])

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'

# caseName = basins.wrapMaster(dataName='basinRef', trainName='first50', batchSize=[
#                              None, 1000], outName='basinRef-first50-opt1')
# slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)

# caseName = basins.wrapMaster(dataName='basinRef', trainName='first50', batchSize=[
#                              None, 1000], varX=usgs.varQ+gridMET.varLst, varY=None, outName='basinRef-first50-opt2')
# slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=24)

# test for small regions

trainLst = ['areaLT10-F50', 'areaGT2500-F50', 'eco0503-F50',
            'eco0902-F50', 'nutr06-F50', 'nutr08-F50']
for trainName in trainLst:
    caseName = basins.wrapMaster(dataName='basinRef', trainName=trainName, batchSize=[
                                 None, 1000], outName='basinRef-{}-opt1'.format(trainName))
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=4)
for trainName in trainLst:
    caseName = basins.wrapMaster(dataName='basinRef', trainName=trainName, batchSize=[
                                 None, 1000], outName='basinRef-{}-opt2'.format(trainName),
                                 varX=usgs.varQ+gridMET.varLst, varY=None)
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=4)
