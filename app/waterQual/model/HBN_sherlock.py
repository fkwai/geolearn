from hydroDL.master import slurm
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins

import pandas as pd
import numpy as np
import os
import time


caseLst = list()
subsetLst = ['Y8090', 'Y0010']
for subset in subsetLst:
    saveName = 'Silica16-{}-opt1'.format(subset)
    caseName = basins.wrapMaster(dataName='Silica16', trainName=subset,
                                 batchSize=[None, 200], outName=saveName)
    caseLst.append(caseName)
    saveName = 'Silica16-{}-opt2'.format(subset)
    caseName = basins.wrapMaster(dataName='Silica16', trainName=subset,
                                 batchSize=[None, 200], varY=None,
                                 varX=usgs.varQ+gridMET.varLst, outName=saveName)
    caseLst.append(caseName)
    saveName = 'Silica16-{}-opt3'.format(subset)
    caseName = basins.wrapMaster(dataName='Silica16', trainName=subset,
                                 batchSize=[None, 200], varY=None, outName=saveName)
    caseLst.append(caseName)
    saveName = 'Silica16-{}-opt4'.format(subset)
    caseName = basins.wrapMaster(dataName='Silica16', trainName=subset,
                                 batchSize=[None, 200], varYC=None, outName=saveName)
    caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=4)

# # predict a variable combination
# subsetLst = ['00618-00955-all-Y8090',
#              '00618-00955-any-Y8090', '00955-Y8090', '00618-Y8090']
# varLst = [['00618', '00955'], ['00618', '00955'], ['00955'], ['00618']]
# for subset, var in zip(subsetLst, varLst):
#     caseName = basins.wrapMaster(
#         dataName='HBN', trainName=subset, batchSize=[None, 200], outName='HBN-'+subset+'-opt1', varYC=var, varX=usgs.varQ+gridMET.varLst, varY=None)
#     caseLst.append(caseName)
#     caseName = basins.wrapMaster(
#         dataName='HBN', trainName=subset, batchSize=[None, 200], outName='HBN-'+subset+'-opt2', varYC=var, varX=gridMET.varLst)
#     caseLst.append(caseName)
#     caseName = basins.wrapMaster(
#         dataName='HBN', trainName=subset, batchSize=[None, 200], outName='HBN-'+subset+'-opt3', varYC=var, varY=None)
#     caseName = basins.wrapMaster(
#         dataName='HBN', trainName=subset, batchSize=[None, 200], outName='HBN-'+subset+'-opt4', varYC=None)
#     caseLst.append(caseName)

# predict a single variable
# codeLst = ['00955', '00915', '00405', '71846', '00410']
# for code in codeLst:
#     saveName = 'HBN-first50-{}'.format(code)
#     caseName = basins.wrapMaster(
#         dataName='HBN', trainName='first50', batchSize=[None, 200],
#         outName=saveName, varYC=[code], varX=usgs.varQ+gridMET.varLst, varY=None)
#     caseLst.append(caseName)


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

# subsetLst = ['{}s-rm'.format(x) for x in ['80', '90', '00', '10']]
# for subset in subsetLst:
#     saveName = 'HBN-{}-opt1'.format(subset)
#     caseName = basins.wrapMaster(dataName='HBN', trainName=subset,
#                                  batchSize=[None, 200], outName=saveName)
#     caseLst.append(caseName)
#     saveName = 'HBN-{}-opt2'.format(subset)
#     caseName = basins.wrapMaster(dataName='HBN', trainName=subset,
#                                  batchSize=[None, 200], varYC=usgs.varC,
#                                  varX=usgs.varQ+gridMET.varLst, outName=saveName)
#     caseLst.append(caseName)
