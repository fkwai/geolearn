import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np

importlib.reload(dbBasin.io)
importlib.reload(dbBasin.dataModel)
importlib.reload(dbBasin)
importlib.reload(basinFull)
importlib.reload(trainBasin)



# load sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

dataName = 'F10'
# dm = dbBasin.DataModelFull.new(dataName, siteNoLst[:10])

dm = dbBasin.DataModelFull(dataName)
dm.saveSubset('f3', siteNoLst[:3])

# get subset
subset = 'f3'
sd = '1979-01-01'
ed = '2010-01-01'
varX = dm.varF
varY = ['runoff']
varXC = dm.varG
varYC = None
varTup = (varX, varXC, varY, varYC)
(x, xc, y, yc) = dm.extractData(varTup, subset, sd, ed)
outName = 'F10B10'
basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                     varY=varY, varXC=varXC, varYC=varYC,
                     sd=sd, ed=ed, subset=subset)
basinFull.trainModel(outName)
