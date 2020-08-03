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
dataName = 'Silica64'
subsetLst = ['00955-Y8090', '00955-Y0010']
codeLst = ['00955']
for subset in subsetLst:
    for hiddenSize in [256, 128, 64, 32]:
        saveName = '{}-{}-h{}-opt1'.format(dataName, subset, hiddenSize)
        caseName = basins.wrapMaster(dataName=dataName, trainName=subset, hiddenSize=hiddenSize,
                                     batchSize=[None, 200], outName=saveName)
        caseLst.append(caseName)
        # saveName = '{}-{}-opt2'.format(dataName, subset)
        # caseName = basins.wrapMaster(dataName=dataName, trainName=subset, hiddenSize=hiddenSize,
        #                              batchSize=[None, 200], varY=None,
        #                              varX=usgs.varQ+gridMET.varLst, outName=saveName)
        # caseLst.append(caseName)
    # saveName = '{}-{}-opt3'.format(dataName, subset)
    # caseName = basins.wrapMaster(dataName=dataName, trainName=subset,
    #                              batchSize=[None, 200], varY=None, outName=saveName)
    # caseLst.append(caseName)
    # saveName = '{}-{}-opt4'.format(dataName, subset)
    # caseName = basins.wrapMaster(dataName=dataName, trainName=subset,
    #                              batchSize=[None, 200], varYC=None, outName=saveName)
    # caseLst.append(caseName)


cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/app/waterQual/model/cmdTrain.py -M {}'
for caseName in caseLst:
    slurm.submitJobGPU(caseName, cmdP.format(caseName), nH=6)
