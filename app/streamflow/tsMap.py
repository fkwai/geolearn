import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
import scipy
from hydroDL.data import dbBasin
from hydroDL.master import basinFull
import os
import pandas as pd
from hydroDL import kPath, utils
import importlib
import time
import numpy as np
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
import matplotlib

dataName = 'Q90ref'
dm = dbBasin.DataModelFull(dataName)
indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]

# global model
nashLst1 = list()
rmseLst1 = list()
corrLst1 = list()
outName = '{}-B10'.format(dataName)
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet='all')
yO, ycO = basinFull.getObs(outName, 'all', DM=dm)
corr1 = utils.stat.calCorr(yP[:indT, :, 0], yO[:indT, :, 0])
corr2 = utils.stat.calCorr(yP[indT:, :, 0], yO[indT:, :, 0])

# tsMap
siteNoLst = dm.siteNoLst
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, corr2, vRange=[0.5, 1], s=16)
    axM.set_title('corr LSTM streamflow')
    figP = plt.figure(figsize=[16, 6])
    figP, axP = plt.subplots(1, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    t = dm.t
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['Obs', 'LSTM']
    axplot.plotTS(axP, t, [yO[:, iP, 0], yP[:, iP, 0]],
                  tBar=tBar, sd=sd, styLst='--', cLst='rb', legLst=legLst)
    axP.legend()

importlib.reload(axplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)
