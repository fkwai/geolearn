
import hydroDL.utils.stat
from scipy import interpolate
import numpy as np
import netCDF4
from hydroDL import kPath
import os
import pandas as pd
import hydroDL.data.cmip.io
import hydroDL.data.gridMET.io
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
varG = 'pr'
varC = 'pr'
func = 'nansum'

# varG = 'tmmx'
# varC = 'tasmax'
# func = 'nanmean'


# read CMIP6
importlib.reload(hydroDL.data.cmip.io)
df = hydroDL.data.cmip.io.walkFile()
mLst = ['MPI-ESM1-2-XR', 'EC-Earth3P-HR', 'EC-Earth3P', 'CNRM-CM6-1-HR']

temp = hydroDL.data.cmip.io.findFile(
    dfFile=df, var=varC, exp='highres-future',
    sd=d2, ed=d3, model=mLst[0])
fileLst = hydroDL.data.cmip.iodf2file(temp)
latC, lonC = hydroDL.data.cmip.io.readCrd(fileLst[0])

for modelName in mLst[1:]:
    temp = hydroDL.data.cmip.io.findFile(
        dfFile=df, var=varC, exp='highres-future',
        sd=d2, ed=d3, model=modelName)
    fileLst = hydroDL.data.cmip.io.df2file(temp)
    latC2, lonC2 = hydroDL.data.cmip.io.readCrd(fileLst[0])
    np.array_equal(latC, latC2)
    np.array_equal(lonC, lonC2)
