import importlib
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import usgs, gageII, gridMET, transform
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('Silica64')
code = '00955'
trainset = 'Y8090'
testset = 'Y0010'
outName = 'Silica64-Y8090-00955-opt1'

siteNoLst = wqData.info['siteNo'].unique()

sd = np.datetime64('1979-01-01')
ed = np.datetime64('2020-01-01')
master = basins.loadMaster(outName)
ep = 500

(varX, varXC, varY, varYC) = (
    master['varX'], master['varXC'], master['varY'], master['varYC'])
(statX, statXC, statY, statYC) = basins.loadStat(outName)
model = basins.loadModel(outName, ep=ep)
tabG = gageII.readData(varLst=varXC, siteNoLst=siteNoLst)
tabG = gageII.updateCode(tabG)

# siteNoF = '07083000'
# siteNoG = '09095500'
tabG.loc[['07083000', '09095500']].T
siteNo = '07083000'
siteNo2 = '09095500'

dfG0 = tabG.loc[siteNo]
dfG2 = tabG.loc[siteNo2]
dfX = waterQuality.readSiteX(
    siteNo, varX, sd=sd, ed=ed, area=None, nFill=5)

fieldLst = ['SLOPE_PCT']
title = ''
for field in fieldLst:
    dfG = tabG.loc[siteNo]
    dfG[field] = dfG2[field]
    title = title+' '+'{}: {}->{}'.format(field, dfG0[field], dfG2[field])

title = 'changed everything but '
fieldLst = ['DRAIN_SQKM', 'SLOPE_PCT', 'FORESTNLCD06', 'GEOL_REEDBUSH_DOM','STREAMS_KM_SQ_KM']
for field in fieldLst:
    dfG = tabG.loc[siteNo2]
    dfG[field] = dfG0[field]
    title = title+field + ', '

dfOut0 = runModel(dfX, dfG0)
dfOut = runModel(dfX, dfG)
fig, ax = plt.subplots(1, 1, figsize=(16, 3))
t = dfOut.index.values.astype('datetime64[D]')
axplot.plotTS(ax, t, [dfOut0['00955'], dfOut['00955']],
              styLst='--', cLst='br', legLst=['original', 'modified'])
ax.set_title(title)
fig.show()


def runModel(dfX, dfG):
    # test model
    xA = np.expand_dims(dfX.values, axis=1)
    xcA = np.expand_dims(
        dfG.values.astype(np.float), axis=0)
    mtdX = wqData.extractVarMtd(varX)
    x = transform.transInAll(xA, mtdX, statLst=statX)
    mtdXC = wqData.extractVarMtd(varXC)
    xc = transform.transInAll(xcA, mtdXC, statLst=statXC)
    yOut = trainTS.testModel(model, x, xc)
    # transfer out
    nt = len(dfX)
    ny = len(varY) if varY is not None else 0
    nyc = len(varYC) if varYC is not None else 0
    yP = np.full([nt, ny+nyc], np.nan)
    yP[:, :ny] = wqData.transOut(yOut[:, 0, :ny], statY, varY)
    yP[:, ny:] = wqData.transOut(yOut[:, 0, ny:], statYC, varYC)
    # save output
    t = dfX.index.values.astype('datetime64[D]')
    colY = [] if varY is None else varY
    colYC = [] if varYC is None else varYC
    dfOut = pd.DataFrame(data=yP, columns=colY+colYC, index=t)
    dfOut.index.name = 'date'
    dfOut = dfOut.reset_index()
    return dfOut
