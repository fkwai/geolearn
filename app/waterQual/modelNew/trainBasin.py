from hydroDL.master import slurm
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins

dataName = 'Silica64'
subset = '00955-Y8090'
saveName = '{}-{}-AgeLSTM'.format(dataName, subset)
caseName = basins.wrapMaster(dataName=dataName, trainName=subset, hiddenSize=512,
                             batchSize=[None, 200], outName=saveName,
                             modelName='AgeLSTM', crit='RmseLoss2D')
basins.trainModelTS(caseName)

