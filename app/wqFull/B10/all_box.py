import importlib
from hydroDL import kPath, utils
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, dbBasin
from hydroDL.post import axplot, figplot
from hydroDL.master import basinFull

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dataNameLst = ['brWN5', 'brDN5']
labelLst = ['QFPRT2C', 'QFPT2C', 'FPRT2QC', 'QT2C']
rhoLst = [365, 10]

dataName = 'brDN5'
labelLst = ['QFPRT2C', 'QFPT2C']
rho = 365
for k, label in enumerate(labelLst):
    outName = '{}-{}-t{}-B10'.format(dataName, label, rho)
    dm = dbBasin.DataModelFull(dataName)
    master = basinFull.loadMaster(outName)
    varY = master['varY']
    testSet = 'all'
    sd = '1982-01-01'
    ed = '2018-12-31'
    ns = len(dm.siteNoLst)
    nc = len(varY)
    yP, ycP = basinFull.testModel(
        outName, DM=dm, batchSize=20, testSet=testSet, ep=100)
    yO, ycO = basinFull.getObs(outName, testSet, DM=dm)

    corrMat = np.ndarray([ns, nc, 2])
    for ic in range(nc):
        indT = np.where(dm.t == np.datetime64('2010-01-01'))[0][0]
        corr1 = utils.stat.calCorr(yP[:indT, :, ic], yO[:indT, :, ic])
        corr2 = utils.stat.calCorr(yP[indT:, :, ic], yO[indT:, :, ic])
        corrMat[:, ic, k] = corr2

dataBox = list()
for ic in range(nc):
    temp = [corrMat[:, ic, 0], corrMat[:, ic, 1]]
    # temp = [rmseMat[:, ic, 0], rmseMat[:, ic, 1]]
    dataBox.append(temp)

labLst1 = ['{}\n{}'.format(usgs.codePdf.loc[code]
                           ['shortName'], code) for code in varY]
labLst2 = ['w/o LAI', 'w/ LAI']
fig = figplot.boxPlot(dataBox, widths=0.5, cLst='br', label1=labLst1, label2=labLst2,
                      figsize=(6, 4), yRange=[0, 1])
fig.show()

# ts random
iP = np.random.randint(ns)
code = '00915'
iC = varY.index(code)
fig, ax = plt.subplots(1, 1)
axplot.plotTS(ax, dm.t, [yP[:, iP, iC], yO[:, iP, iC]], styLst='-*', cLst='bk')
fig.show()
