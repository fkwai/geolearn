

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
from hydroDL.app.waterQuality import WRTDS
import os
from hydroDL import kPath, utils
import numpy as np

dataName = 'G200Norm'
outName = dataName
trainSet = 'rmRT20'
testSet = 'pkRT20'
# testSet = 'all'

DF = dbBasin.DataFrameBasin(outName)
code = usgs.newC
yW = WRTDS.testWRTDS(dataName, trainSet, testSet, usgs.newC)

dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, testSet)
np.save(os.path.join(dirRoot,fileName), yW)
