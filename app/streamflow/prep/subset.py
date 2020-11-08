import os
import time
import pandas as pd
import numpy as np
import json
from hydroDL import kPath, utils
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import basins
from hydroDL.app import waterQuality

caseName = 'Q90'
saveFolder = os.path.join(kPath.dirWQ, 'trainDataFull', caseName)
with open(os.path.join(saveFolder, 'info')+'.json', 'r') as fp:
    info = json.load(fp)
tR = pd.date_range(np.datetime64(info['sd']), np.datetime64(info['ed']))
siteNoLst = info['siteNoLst']

# before 2010
subName = 'B10'
subMat = np.zeros([len(tR), len(siteNoLst)], dtype=bool)
subMat[tR < np.datetime64('2010-01-01'), :] = True
np.save(os.path.join(saveFolder, 'subset_'+subName), subMat)

# after 2010
subName = 'A10'
subMat = np.zeros([len(tR), len(siteNoLst)], dtype=bool)
subMat[tR >= np.datetime64('2010-01-01'), :] = True
np.save(os.path.join(saveFolder, 'subset_'+subName), subMat)
