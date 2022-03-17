
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

# dataName = 'G200N'
# labelLst = ['QFPRT2C', 'FPRT2C', 'FPRT2QC', 'QFPT2C', 'QFRT2C']

dataName = 'G200'
labelLst = ['QFPRT2C']

trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']
DF = dbBasin.DataFrameBasin(dataName)

for label in labelLst:
    for trainSet, testSet in zip(trainLst, testLst):
        outName = '{}-{}-{}'.format(dataName, label, trainSet)
        print(outName)
        yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=500)
