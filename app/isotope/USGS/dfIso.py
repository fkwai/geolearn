import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
from hydroDL.data import gageII, usgs, gridMET, dbBasin
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os

# site inventory
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLstIso = ['82085', '82745', '82082']
t0 = time.time()
sd = pd.datetime(1982, 1, 1)
siteNoLst = list()
countLst = list()
dfLst = list()
for i, siteNo in enumerate(siteNoLstAll):
    print(i)
    dfC, dfCF = usgs.readSample(
        siteNo, codeLst=codeLstIso, flag=2, startDate=sd, csv=True)
    dfC = dfC.dropna(how='all')
    if len(dfC) > 0:
        dfLst.append(dfC)
        siteNoLst.append(siteNo)
        countLst.append(len(dfC))

# plot selected sites
countThe = 50
siteNoSel = list()
countSel = list()
dfSel = list()
for i, siteNo in enumerate(siteNoLst):
    if countLst[i] > countThe:
        dfSel.append(dfLst[i])
        siteNoSel.append(siteNoLst[i])
        countSel.append(countLst[i])
len(siteNoSel)

# create a DB for later use
dataName = 'isoAll'
codeLst = codeLstIso+usgs.newC
DF = dbBasin.DataFrameBasin.new(
    dataName, siteNoLst, varC=codeLst, varG=gageII.varLstEx)
# not working need to wrap up