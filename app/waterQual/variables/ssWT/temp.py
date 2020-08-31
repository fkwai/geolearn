from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


freq = 'W'
area = None
nFill = 5
sd = np.datetime64('1979-01-01')
ed = np.datetime64('2019-12-31')
varLst = ['00940', '00060', 'pr', 'ph']

siteNo = '0143400680'

if freq == 'D':
    td = pd.date_range(sd, ed)
    tr = pd.date_range(sd, ed)
elif freq == 'W':
    if ed > np.datetime64('2019-12-30'):
        ed = np.datetime64('2019-12-30')
    tr = pd.date_range(sd, ed, freq='W-TUE')
    ed = tr[-1]
    td = pd.date_range(sd, ed)
# read data
varC = list(set(varLst).intersection(usgs.varC))
varQ = list(set(varLst).intersection(usgs.varQ))
varF = list(set(varLst).intersection(gridMET.varLst))
varP = list(set(varLst).intersection(ntn.varLst))

dfD = pd.DataFrame({'date': td}).set_index('date')
if len(varC) > 0:
    dfC = usgs.readSample(siteNo, codeLst=varC, startDate=sd)
    dfD = dfD.join(dfC)
if len(varQ) > 0:
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
if len(varF) > 0:
    dfF = gridMET.readBasin(siteNo, varLst=varF)
if len(varP) > 0:
    dfP = ntn.readBasin(siteNo, varLst=varP, freq=freq)


# extract data
dfF = gridMET.readBasin(siteNo)
if '00060' in varX or 'runoff' in varX:
    dfQ = usgs.readStreamflow(siteNo, startDate=sd)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    if 'runoff' in varX:
        if area is None:
            tabArea = gageII.readData(
                varLst=['DRAIN_SQKM'], siteNoLst=[siteNo])
            area = tabArea['DRAIN_SQKM'].values[0]
        dfQ['runoff'] = calRunoffArea(dfQ['00060'], area)
    dfX = dfX.join(dfQ)
dfX = dfX.join(dfF)
dfX = dfX[varX]
dfX = dfX.interpolate(limit=nFill, limit_direction='both')
