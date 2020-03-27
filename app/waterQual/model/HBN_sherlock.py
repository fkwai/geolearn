from hydroDL.master import slurm
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
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


caseLst = list()
# for opt in [1, 2, 3, 4]:
#     for trainName, trainStr in zip(['first80', 'first80-rm2'], ['', '-rm2']):
#         saveName = 'HBN-opt'+str(opt)+trainStr
#         caseName = basins.wrapMaster('HBN', trainName, batchSize=[
#                                      None, 200], optQ=opt, outName=saveName)
#         caseLst.append(caseName)

# predict q only
# caseLst = list()
# saveName = 'HBN-first50-q'
# caseName = basins.wrapMaster(
#     dataName='HBN', trainName='first50', batchSize=[None, 200],
#     outName=saveName, varYC=None)
# caseLst.append(caseName)

# saveName = 'HBN-first80-q'
# caseName = basins.wrapMaster(
#     dataName='HBN', trainName='first80', batchSize=[None, 200],
#     outName=saveName, varYC=None)
# caseLst.append(caseName)


# 30 day
# caseLst=list()
# saveName = 'HBN-30d-first50-opt1'
# caseName = basins.wrapMaster(dataName='HBN-30d', trainName='first50',
#                              batchSize=[None, 200], outName=saveName)
# caseLst.append(caseName)

# saveName = 'HBN-30d-first50-opt2'
# caseName = basins.wrapMaster(dataName='HBN-30d', trainName='first50',
#                              batchSize=[None, 200], varYC=usgs.varC,
#                              varX=usgs.varQ+gridMET.varLst, outName=saveName)
# caseLst.append(caseName)

# group only
# codePdf = usgs.codePdf
# groupLst = codePdf.group.unique().tolist()
# groupLst.reverse()
# for group in groupLst:
#     # predict a group of c only
#     codeLst = codePdf[codePdf.group == group].index.tolist()
#     saveName = 'HBN-30d-first50-opt1-'+group
#     caseName = basins.wrapMaster(
#         dataName='HBN-30d', trainName='first50', batchSize=[None, 200],
#         outName=saveName, varYC=codeLst)
#     caseLst.append(caseName)

#     saveName = 'HBN-30d-first50-opt2-'+group
#     caseName = basins.wrapMaster(
#         dataName='HBN-30d', trainName='first50', batchSize=[None, 200],
#         outName=saveName, varYC=codeLst, varX=usgs.varQ+gridMET.varLst)
#     caseLst.append(caseName)

subsetLst = ['{}s-rm'.format(x) for x in ['80', '90', '00', '10']]
for subset in subsetLst:
    saveName = 'HBN-{}-opt1'.format(subset)
    caseName = basins.wrapMaster(dataName='HBN', trainName=subset,
                                 batchSize=[None, 200], outName=saveName)
    caseLst.append(caseName)
    saveName = 'HBN-{}-opt2'.format(subset)
    caseName = basins.wrapMaster(dataName='HBN', trainName=subset,
                                 batchSize=[None, 200], varYC=usgs.varC,
                                 varX=usgs.varQ+gridMET.varLst, outName=saveName)
    caseLst.append(caseName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=4)
