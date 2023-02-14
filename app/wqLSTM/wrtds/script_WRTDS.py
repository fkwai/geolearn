

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
from hydroDL.app.waterQuality import WRTDS
import os
from hydroDL import kPath, utils
import numpy as np

dataName = 'G200'
trainLst = ['rmR20', 'rmL20', 'rmRT20', 'rmYr5', 'B10']
testLst = ['pkR20', 'pkL20', 'pkRT20', 'pkYr5', 'A10']
# testSet = 'all'

# trainLst = ['B10']
# testLst = ['A10']
DF = dbBasin.DataFrameBasin(dataName)

for trainSet, testSet in zip(trainLst, testLst):
    # testSet = 'all'
    yW = WRTDS.testWRTDS(dataName, trainSet, testSet, usgs.varC)
    dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
    fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
np.savez_compressed(os.path.join(dirRoot, fileName), yW=yW)
