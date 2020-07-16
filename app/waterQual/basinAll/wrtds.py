import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot
from sklearn.linear_model import LinearRegression


import torch
import os
import json
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

siteNo = '01013500'
startDate = pd.datetime(1979, 1, 1)
endDate = pd.datetime(2020, 1, 1)

dfC, dfCF = usgs.readSample(
    siteNo, codeLst=usgs.varC, startDate=startDate, flag=2)
dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
dfX = pd.DataFrame({'date': dfC.index}).set_index('date')
dfY = np.log(dfC+0.01)
dfX = dfX.join(np.log(dfQ['00060_00003']+0.01)).rename(
    columns={'00060_00003': 'logQ'})
yr = dfX.index.year.values
t = yr+dfX.index.dayofyear.values/365
dfX['t'] = t-1979
dfX['sinT'] = np.sin(2*np.pi*t)
dfX['cosT'] = np.cos(2*np.pi*t)

ctR = pd.date_range(startDate, endDate)
dfXP = pd.DataFrame({'date': ctR}).set_index('date')
dfYP = pd.DataFrame({'date': ctR}).set_index('date')

dfXP = dfXP.join(np.log(dfQ['00060_00003']+0.01)).rename(
    columns={'00060_00003': 'logQ'})
yr = dfXP.index.year.values
t = yr+dfXP.index.dayofyear.values/365
dfXP['t'] = t-1979
dfXP['sinT'] = np.sin(2*np.pi*t)
dfXP['cosT'] = np.cos(2*np.pi*t)

code = '00681'
[xx, yy], iv = utils.rmNan([dfX.values, dfY[code].values])
lrModel = LinearRegression()
lrModel.fit(xx, yy)
b = dfXP.isna().any(axis=1)
yp = lrModel.predict(dfXP[~b].values)
yp=
dfYP[code]=np.nan
dfYP.at[dfXP[~b].index,code]=yp