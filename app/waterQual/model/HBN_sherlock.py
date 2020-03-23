from hydroDL.master import slurm
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time

# all gages
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfHBN = pd.read_csv(os.path.join(kPath.dirData, 'USGS', 'inventory', 'HBN.csv'), dtype={
    'siteNo': str}).set_index('siteNo')
siteNoHBN = [siteNo for siteNo in dfHBN.index.tolist()
             if siteNo in siteNoLstAll]

# wrap up data
if not waterQuality.exist('HBN'):
    wqData = waterQuality.DataModelWQ.new('HBN', siteNoHBN)
else:
    wqData = waterQuality.DataModelWQ('HBN')
if 'first80-rm2' not in wqData.subset.keys():
    ind = wqData.subset['first80']
    indRm = wqData.indByComb(['00010', '00095'])
    indTrain = np.setdiff1d(ind, indRm)
    wqData.saveSubset('first80-rm2', indTrain)


caseLst = list()
# for opt in [1, 2, 3, 4]:
#     for trainName, trainStr in zip(['first80', 'first80-rm2'], ['', '-rm2']):
#         saveName = 'HBN-opt'+str(opt)+trainStr
#         caseName = basins.wrapMaster('HBN', trainName, batchSize=[
#                                      None, 200], optQ=opt, outName=saveName)
#         caseLst.append(caseName)

saveName = 'HBN-first50-q'
caseName = basins.wrapMaster(
    'HBN', 'first50', batchSize=[None, 200], outName=saveName,
    varX=)
caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=4)
