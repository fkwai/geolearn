
import matplotlib
import datetime
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from hydroDL import utils
import os
from hydroDL.model import trainBasin, crit, waterNetTest
from hydroDL.data import dbBasin, gageII
import numpy as np
import torch
import pandas as pd
from hydroDL.model import waterNetTest
from hydroDL.master import basinFull
import importlib

trainSet = 'WYB09'
testSet = 'WYA09'
dataName = 'QN90ref'
# dataName = 'Q95ref'
wnName = 'WaterNet0119'
epWN = 500
epLSTM = 500
modelFile = '{}-{}-ep{}'.format(wnName, dataName, epWN)
lstmOutName = '{}-{}'.format(dataName, trainSet)
dirOut = r'C:\Users\geofk\work\waterQuality\waterNet\outTemp'

# data
DF = dbBasin.DataFrameBasin(dataName)
DM = dbBasin.DataModelBasin(DF, varY=['runoff'],
                            subset=testSet, varX=None, varXC=None, varYC=None)
DM.trans(mtdY=['skip'])

dataTup = DM.getData()
yT = dataTup[2][:, :, 0]

# waterNet
nr = 5
outName = 'ts{}-{}.npz'.format(modelFile, testSet)
outFile = os.path.join(dirOut, outName)
outW = np.load(outFile)
outName = 'gate{}'.format(modelFile)
outFile = os.path.join(dirOut, outName+'.npz')
outP = np.load(outFile)
yP = outW['yP']
QpP = outW['QpP']
QsP = outW['QsP']
QgP = outW['QgP']
SfP = outW['SfP']
SsP = outW['SsP']
SgP = outW['SgP']
qb = outP['qb']


# LSTM
lstmOutName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    lstmOutName, DF=DF, testSet=testSet, reTest=False, ep=epLSTM)
yL = yL[:, :, 0]

lat, lon = DF.getGeo()

nash1 = utils.stat.calNash(yP, yT[nr-1:, :])
nash2 = utils.stat.calNash(yL[nr-1:, :], yT[nr-1:, :])
corr1 = utils.stat.calCorr(yP, yT[nr-1:, :])
corr2 = utils.stat.calCorr(yL[nr-1:, :], yT[nr-1:, :])
bias1 = utils.stat.calBiasR(yP, yT[nr-1:, :])
bias2 = utils.stat.calBiasR(yL, yT)
var1 = utils.stat.calVarR(yP, yT[nr-1:, :])
var2 = utils.stat.calVarR(yL, yT)
lat, lon = DF.getGeo()
t = DF.getT(testSet)

importlib.reload(figplot)


def funcM():
    figM = plt.figure()
    gsM = gridspec.GridSpec(3, 1)
    axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, nash1)
    axM0.set_title('waterNet Nash')
    axM1 = mapplot.mapPoint(figM, gsM[1, 0], lat, lon, nash2)
    axM1.set_title('LSTM Nash')
    axM2 = mapplot.mapPoint(figM, gsM[2, 0], lat, lon, nash2-nash1)
    axM2.set_title('LSTM - waterNet Nash')
    axM = np.array([axM0, axM1, axM2])
    figP, axP = plt.subplots(4, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    siteNo = DF.siteNoLst[iP]
    legLst = ['obs',
              'waterNet {:.2f} {:.2f}'.format(nash1[iP], corr1[iP]),
              'LSTM {:.2f} {:.2f}'.format(nash2[iP], corr2[iP])
              ]
    axplot.plotTS(axP[0], t[nr-1:], [yT[nr-1:, iP], yP[:, iP], yL[nr-1:, iP]],
                  lineW=[2, 1, 1], cLst='krb', legLst=legLst)
    strTitle = ('{}'.format(siteNo))
    axP[0].set_title(strTitle)
    axP[1].plot(t, SfP[:, iP, :])
    axP[2].plot(t, SsP[:, iP, :])
    axP[3].plot(t, SgP[:, iP, :])


figM, figP = figplot.clickMap(funcM, funcP)


dirFig = r'C:\Users\geofk\work\Presentation\AGU2022\posterFig'
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
legLst = ['obs', 'waterNet', 'LSTM']
iP = 375
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(5, 1)
ax4 = fig.add_subplot(gs[4])
ax1 = fig.add_subplot(gs[0:2], sharex=ax4)
ax2 = fig.add_subplot(gs[2], sharex=ax4)
ax3 = fig.add_subplot(gs[3], sharex=ax4)
legLst = ['obs', 'waterNet', 'LSTM']
axplot.plotTS(ax1, t[nr-1:], [yT[nr-1:, iP], yP[:, iP], yL[nr-1:, iP]],
              lineW=[2, 1, 1], cLst='krb', legLst=legLst)
ax2.plot(t, SfP[:, iP, :])
ax3.plot(t, SsP[:, iP, :])
ax4.plot(t, SgP[:, iP, :])
ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax3.xaxis.set_visible(False)
ax1.set_ylim([0, 5])
fig.subplots_adjust(hspace=0)
fig.show()
ax1.set_xlim([datetime.date(2010, 3, 1), datetime.date(2010, 9, 1)])

fig.savefig(os.path.join(dirFig, 'ts3.svg'))


# ax1.set_xlim([datetime.date(2014, 1, 1), datetime.date(2017, 1, 1)])


iP = 117
fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(5, 1)
ax4 = fig.add_subplot(gs[4])
ax1 = fig.add_subplot(gs[0:2], sharex=ax4)
ax2 = fig.add_subplot(gs[2], sharex=ax4)
ax3 = fig.add_subplot(gs[3], sharex=ax4)
legLst = ['obs', 'waterNet', 'LSTM']
axplot.plotTS(ax1, t[nr-1:], [yT[nr-1:, iP], yP[:, iP], yL[nr-1:, iP]],
              lineW=[2, 1, 1], cLst='krb', legLst=legLst)
ax2.plot(t, SfP[:, iP, :])
ax3.plot(t, SsP[:, iP, :])
ax4.plot(t, SgP[:, iP, :])
ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax3.xaxis.set_visible(False)
fig.subplots_adjust(hspace=0)
ax1.set_ylim([0, 40])
fig.show()
fig.savefig(os.path.join(dirFig, 'ts2.svg'))

DF.siteNoLst[375]
DF.siteNoLst[117]

nash1[375]
nash2[375]
nash1[117]
nash2[117]