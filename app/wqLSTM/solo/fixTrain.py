import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
from hydroDL.master import basinFull
from hydroDL.master import slurm

codeLst = usgs.varC
labelLst = ['FT2QC', 'QFT2C', 'QT2C']
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
ep = 500
ep1=20
errLst1 = list()
errLst2 = list()
errLst3 = list()

for code in codeLst:
    dataName = '{}-{}'.format(code, 'B200')
    # DF = dbBasin.DataFrameBasin(dataName)
    # local model
    for label in labelLst:
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        dictMaster = basinFull.loadMaster(outName)
        outFolder = basinFull.nameFolder(outName)
        if not os.path.exists(os.path.join(outFolder,'modelState_ep{}'.format(ep1))):
            errLst1.append(outName)        
        else:
            if not os.path.exists(os.path.join(outFolder,'modelState_ep{}'.format(ep))):
                errLst2.append(outName)
        if not os.path.exists(os.path.join(outFolder,'master.json')):
            errLst3.append(outName)

cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
for outName in errLst1:
    slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)

  