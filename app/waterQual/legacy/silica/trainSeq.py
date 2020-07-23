
from hydroDL import kPath
from hydroDL.app import waterQuality, waterQuality2
from hydroDL.master import basins


# import importlib
from hydroDL.master import slurm
import numpy as np


# importlib.reload(waterQuality2)

# wqData = waterQuality.DataModelWQ('Silica64')
# siteNoLst = wqData.siteNoLst
# if not waterQuality.exist('Silica64Seq'):
#     wqData = waterQuality2.DataModelWQ.new('Silica64Seq', siteNoLst)
# importlib.reload(waterQuality2)
# wqData = waterQuality2.DataModelWQ('Silica64Seq')
temp = waterQuality.DataModelWQ('Silica64')
siteNoLst = temp.siteNoLst
# wqData = waterQuality2.DataModelWQ.new('Silica64Seq', siteNoLst)

wqData = waterQuality2.DataModelWQ('Silica64Seq')

# subset only have silica
code = '00955'
ic = wqData.varQ.index(code)
indC = np.where(~np.isnan(wqData.q[-1,:, ic]))[0]
wqData.saveSubset(code, indC)
indYr1 = waterQuality.indYr(wqData.info.iloc[indC], yrLst=[1979, 2000])[0]
wqData.saveSubset('{}-Y8090'.format(code), indYr1)
indYr2 = waterQuality.indYr(wqData.info.iloc[indC], yrLst=[2000, 2020])[0]
wqData.saveSubset('{}-Y0010'.format(code), indYr2)

saveName = 'Silica64Seq-Y8090'
caseName = basins.wrapMaster(dataName='Silica64Seq', trainName='00955-Y8090',
                             batchSize=[None, 200], varY=['00060','00955'], varYC=None,
                             outName=saveName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=6)
