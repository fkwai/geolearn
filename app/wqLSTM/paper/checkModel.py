
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
# label = 'FPRT2QC'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
outFolder = basinFull.nameFolder(outName)
dictP = basinFull.loadMaster(outName)
dictVar = {k: dictP[k]
           for k in ('varX', 'varXC', 'varY', 'varYC')}
DM = dbBasin.DataModelBasin(DF, subset=testSet, **dictVar)
DM.loadStat(outFolder)
dataTup = DM.getData()
# dataTup = trainBasin.dealNaN(dataTup, dictP['optNaN'])
model = basinFull.defineModel(dataTup, dictP)
model = basinFull.loadModelState(outName, 500, model)
