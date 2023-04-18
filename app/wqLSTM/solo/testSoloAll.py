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
ep1 = 20
errLst1 = list()
errLst2 = list()
errLst3 = list()
outLst = list()

for code in codeLst:
    dataName = '{}-{}'.format(code, 'B200')
    # DF = dbBasin.DataFrameBasin(dataName)
    # local model
    for label in labelLst:
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        dictMaster = basinFull.loadMaster(outName)
        outFolder = basinFull.nameFolder(outName)
        if not os.path.exists(os.path.join(outFolder, 'modelState_ep{}'.format(ep1))):
            errLst1.append(outName)
        else:
            if not os.path.exists(
                os.path.join(outFolder, 'modelState_ep{}'.format(ep))
            ):
                errLst2.append(outName)
            else:
                outLst.append(outName)
        if not os.path.exists(os.path.join(outFolder, 'master.json')):
            errLst3.append(outName)

for outName in outLst:
    yP1, ycP1 = basinFull.testModel(outName, testSet=trainSet, ep=ep)
    yP2, ycP2 = basinFull.testModel(outName, testSet=testSet, ep=ep)
