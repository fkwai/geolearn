from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, dbBasin
from hydroDL.master import slurm
from hydroDL import kPath
from hydroDL.master import basinFull
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'weathering'
codeSel = ['00915', '00925', '00930', '00935', '00940', '00945', '00955']
rho = 365
label = 'QFPRT2C'
# label = 'FPR2QC'
outName = '{}-{}-t{}-B10'.format(dataName, label, rho)
dm = dbBasin.DataModelFull(dataName)
testSet = 'all'
yP, ycP = basinFull.testModel(
    outName, DM=dm, batchSize=20, testSet=testSet, ep=100)
yO = dm.extractVarT(codeSel)
siteNoLst = dm.siteNoLst


t0 = np.datetime64('1982-01-01', 'D')
t1 = np.datetime64('2010-01-01', 'D')
t2 = np.datetime64('2018-12-31', 'D')
indT0 = np.where(dm.t == t0)[0][0]
indT1 = np.where(dm.t == t1)[0][0]
indT2 = np.where(dm.t == t2)[0][0]

# load WRTDS
yW = np.ndarray(yO.shape)
for k, siteNo in enumerate(siteNoLst):
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-D', 'weathering-B10')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    df.index = df.index.values.astype('datetime64[D]')
    ind = (df.index >= t0) & (df.index <= t2)
    yW[:, k, :] = df.loc[ind][codeSel].values

# calculate err
errL = np.ndarray([len(siteNoLst), len(codeSel), 2])
errW = np.ndarray([len(siteNoLst), len(codeSel), 2])
for k, siteNo in enumerate(siteNoLst):
    errL[k, :, 0] = utils.stat.calCorr(
        yP[indT0:indT1, k, :], yO[indT0:indT1, k, :])
    errL[k, :, 1] = utils.stat.calCorr(
        yP[indT1:indT2, k, :], yO[indT1:indT2, k, :])
    errW[k, :, 0] = utils.stat.calCorr(
        yW[indT0:indT1, k, :], yO[indT0:indT1, k, :])
    errW[k, :, 1] = utils.stat.calCorr(
        yW[indT1:indT2, k, :], yO[indT1:indT2, k, :])


dfCrd = gageII.readData(siteNoLst=siteNoLst, varLst=[
                        'DRAIN_SQKM', 'LNG_GAGE', 'LAT_GAGE'])
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
area = dfCrd['DRAIN_SQKM'].values
nc = len(codeSel)


def funcM():
    figM, axM = plt.subplots(2, 1, figsize=(6, 4))
    axplot.mapPoint(axM[0], lat, lon, np.mean(
        errL[:, :, 1], axis=1), s=16, cb=True)
    axplot.mapPoint(axM[1], lat, lon, np.mean(
        errW[:, :, 1], axis=1), s=16, cb=True)
    figP, axP = plt.subplots(nc, 1, figsize=(12, 8))
    figP.subplots_adjust(hspace=0)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    siteNo = siteNoLst[iP]
    labelLst = list()
    for ic, code in enumerate(codeSel):
        shortName = usgs.codePdf.loc[code]['shortName']
        temp = '{} {} LSTM [{:.2f} {:.2f}], WRTDS [{:.2f} {:.2f}]'.format(
            code, shortName, errL[iP, ic, 0], errL[iP, ic, 1],
            errW[iP, ic, 0], errW[iP, ic, 1])
        labelLst.append(temp)
    t = dm.t
    dataPlot = [yW[:, iP, :], yP[:, iP, :], yO[:, iP, :]]
    # t = dm.t[indT1:indT2]
    # dataPlot = [yW[indT1:indT2, iP, :],
    #             yP[indT1:indT2, iP, :], yO[indT1:indT2, iP, :]]
    axplot.multiTS(axP, t, dataPlot, tBar=t1, labelLst=labelLst, cLst='brk')
    figP.suptitle('{} {}'.format(siteNo, area[iP]))


figM, figP = figplot.clickMap(funcM, funcP)
