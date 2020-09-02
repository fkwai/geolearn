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

# read data
td = pd.date_range(sd, ed)
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
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
if len(varF) > 0:
    dfF = gridMET.readBasin(siteNo, varLst=varF)
if len(varP) > 0:
    dfP = ntn.readBasin(siteNo, varLst=varP, freq=freq)

# extract data
dfD = pd.DataFrame({'date': td}).set_index('date')
if 'runoff' in varLst:
    if area is None:
        tabArea = gageII.readData(
            varLst=['DRAIN_SQKM'], siteNoLst=[siteNo])
        area = tabArea['DRAIN_SQKM'].values[0]
    dfQ['runoff'] = waterQuality.calRunoffArea(dfQ['00060'], area)
dfD = dfD.join(dfQ)
dfD = dfD.join(dfF)
dfD = dfD.join(dfC)
dfD = dfD.join(dfP)
dfD = dfD[varLst]
dfD = dfD.interpolate(limit=nFill, limit_direction='both')

if freq == 'D':
    return dfD
elif freq == 'W':
    dfW = dfD.resample('W-TUE').mean()
    return dfW
