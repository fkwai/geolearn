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
dataName = 'CaO49'
subsetLst = ['CaO-Y8090', 'CaO-Y0010']
codeLst = ['00300', '00915']
for subset in subsetLst:
    saveName = '{}-{}-opt1'.format(dataName, subset)
    caseName = basins.wrapMaster(dataName=dataName, trainName=subset,
                                 batchSize=[None, 200], outName=saveName, varYC=codeLst)
    caseLst.append(caseName)
    saveName = '{}-{}-opt2'.format(dataName, subset)
    caseName = basins.wrapMaster(dataName=dataName, trainName=subset,
                                 batchSize=[None, 200], varY=None,
                                 varX=usgs.varQ+gridMET.varLst, outName=saveName, varYC=codeLst)
    caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=12)
