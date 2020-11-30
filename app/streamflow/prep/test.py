import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np

"""
instead of saving time series by rho, save the full time series here. 
f and q will be saved in full matirx
c will saved in sparse matrix 
"""

# load sites
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteSel', 'Q90ref')
siteNoLst = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()

dataName = 'Q90ref'
# dm = dbBasin.DataModelFull.new(dataName, siteNoLst)
dm = dbBasin.DataModelFull(dataName)

varX = dm.varF
varY = ['runoff']
varXC = dm.varG
varYC = None
sd = '1979-01-01'
ed = '2010-01-01'

outName = '{}-B10'.format(dataName)
basinFull.wrapMaster(outName=outName, dataName=dataName, varX=varX,
                     varY=varY, varXC=varXC, varYC=varYC,
                     sd=sd, ed=ed)
basinFull.trainModel(outName)
