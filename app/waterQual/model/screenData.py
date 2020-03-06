import json
import matplotlib.pyplot as plt
from hydroDL.data import usgs, gageII, gridMET
from hydroDL import kPath
from hydroDL.app import waterQuality

import pandas as pd
import numpy as np
import os
import time

# list of site - generate from checkCQ.py, 5978 sites in total
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

# load all data - wrap all data takes 2hrs, 5835 sites left
siteNoLst = siteNoLstAll
caseName = 'basinAll'
wqData = waterQuality.DataModelWQ(caseName)

indTrain, indTest = wqData.indByRatio(0.8)
indCount = wqData.indByCount(20)
indComb = wqData.indByComb(['00010', '00095'])
indTrain = np.setdiff1d(indTrain, indCount)
indTest = np.setdiff1d(indTest, indCount)
indTrainRmComb = np.setdiff1d(indTrain, indComb)
dictSubset = dict(train=indTrain, test=indTest, trainRmComb=indTrainRmComb)
wqData.saveSubset(dictSubset)
