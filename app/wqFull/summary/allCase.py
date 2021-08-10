
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

dataNameLst = ['G200N', 'G200']
labelLst = ['FPRT2QC', 'QFPRT2C', 'QFRT2C', 'QFPT2C', 'QT2C']
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']

# quick scan
dirModel = r'C:\Users\geofk\work\waterQuality\modelFull'
for dataName in dataNameLst:
    for label in labelLst:
        for trainSet in trainLst:
            outName = '{}-{}-{}'.format(dataName, label, trainSet)
            fileName = os.path.join(dirModel, outName, 'modelState_ep500')
            if not os.path.isfile(fileName):
                print(outName)
