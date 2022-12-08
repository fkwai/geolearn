
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy
from hydroDL.data import dbBasin, gageII, camels, gridMET, GLASS


countFile = os.path.join(kPath.dirUSGS, 'streamflow', 'countYr.csv')
dfCount = pd.read_csv(countFile, dtype={'siteNo': str}).set_index('siteNo')
sd = pd.datetime(1982, 1, 1)
ed = pd.datetime(2018, 12, 31)
nt = len(pd.date_range(sd, ed))
yrLst = list(range(1982, 2019))
yrLstStr = [str(yr) for yr in yrLst]

siteNoLst = dfCount.index.tolist()
countYr = dfCount[yrLstStr].values
count = np.sum(countYr, axis=1)
b1 = count > nt*0.95
dfG = gageII.readData(siteNoLst=siteNoLst, varLst=['CLASS'])
b2 = dfG.values.flatten() == 'Ref'
ind = np.where(b1*b2)[0]
sLstRef = [siteNoLst[k] for k in ind]
ind = np.where(b1)[0]
sLst = [siteNoLst[k] for k in ind]

dataName = 'Q95ref'
DF = dbBasin.DataFrameBasin.new(
    dataName, sLstRef, varC=[], varF=gridMET.varLst+['LAI'], varG=gageII.varLstEx)
DF.saveSubset('WYB09', sd='1982-01-01', ed='2009-10-01')
DF.saveSubset('WYA09', sd='2009-10-01', ed='2018-12-31')
DF.saveSubset('WYall', sd='1982-01-01', ed='2018-12-31')
