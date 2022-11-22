
import hydroDL.utils.stat
from scipy import interpolate
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
import hydroDL.data.daymet.io

import importlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from hydroDL.post import mapplot, axplot, figplot
import hydroDL.utils.ts
importlib.reload(hydroDL.data)

# set dates
d1 = np.datetime64('2010-01-01')
d2 = np.datetime64('2015-01-01')
d3 = np.datetime64('2020-01-01')
y1, y2, y3 = [d.astype(object).year for d in [d1, d2, d3]]
latR = [25, 50]
lonR = [-125, -65]
varG = 'prcp'
varC = 'pr'
func = 'nansum'

site1=hydroDL.data.daymet.io.readVal(varG,y1,y2)
site2=hydroDL.data.daymet.io.readVal(varG,y2,y3)

# read CMIP6
importlib.reload(hydroDL.data.cmip.io)
df = hydroDL.data.cmip.io.walkFile()
mLst = ['MPI-ESM1-2-XR', 'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']
rF1Lst = list()
rF2Lst = list()
rM1Lst = list()
rM2Lst = list()
rA1Lst = list()
rA2Lst = list()
# for modelName in mLst:

modelName=mLst=0
data1, latC1, lonC1, tC1 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var=varC, exp='hist-1950', latR=latR, lonR=lonR,
    sd=d1, ed=d2, model=modelName)

data2, latC2, lonC2, tC2 = hydroDL.data.cmip.io.readCMIP(
    dfFile=df, var=varC, exp='highres-future', latR=latR, lonR=lonR,
    sd=d2, ed=d3, model=modelName)

# check consistance
latC = latC1 if np.array_equal(latC1, latC2) else Exception('latC')
lonC = lonC1 if np.array_equal(lonC1, lonC2) else Exception('lonC')


    rF1 = hydroDL.utils.stat.gridCorrT(grid1, data1)
    rF2 = hydroDL.utils.stat.gridCorrT(grid2, data2)

    gridM1, _ = hydroDL.utils.ts.data2Monthly(grid1, t1, func=func)
    dataM1, _ = hydroDL.utils.ts.data2Monthly(data1, t1, func=func)
    gridM2, _ = hydroDL.utils.ts.data2Monthly(grid2, t2, func=func)
    dataM2, _ = hydroDL.utils.ts.data2Monthly(data2, t2, func=func)
    rM1 = hydroDL.utils.stat.gridCorrT(gridM1, dataM1)
    rM2 = hydroDL.utils.stat.gridCorrT(gridM2, dataM2)

    gridA1, _ = hydroDL.utils.ts.data2Climate(grid1, t1, func=func)
    dataA1, _ = hydroDL.utils.ts.data2Climate(data1, t1, func=func)
    gridA2, _ = hydroDL.utils.ts.data2Climate(grid2, t2, func=func)
    dataA2, _ = hydroDL.utils.ts.data2Climate(data2, t2, func=func)
    rA1 = hydroDL.utils.stat.gridCorrT(gridA1, dataA1)
    rA2 = hydroDL.utils.stat.gridCorrT(gridA2, dataA2)

    rF1Lst.append(rF1)
    rF2Lst.append(rF2)
    rM1Lst.append(rM1)
    rM2Lst.append(rM2)
    rA1Lst.append(rA1)
    rA2Lst.append(rA2)

fig, ax = figplot.boxPlot([rF1Lst, rF2Lst], widths=0.5, cLst='rgbkm',
                          label2=mLst, label1=['hist', 'future'],
                          figsize=(4, 4))
fig.suptitle('dailyly correlation')
fig.show()


fig, ax = figplot.boxPlot([rM1Lst, rM2Lst], widths=0.5, cLst='rgbkm',
                          label2=mLst, label1=['hist', 'future'],
                          figsize=(4, 4))
fig.suptitle('monthly correlation')
fig.show()

fig, ax = figplot.boxPlot([rA1Lst, rA2Lst], widths=0.5, cLst='rgbkm',
                          label2=mLst, label1=['hist', 'future'],
                          figsize=(4, 4))
fig.suptitle('climatology correlation')
fig.show()
