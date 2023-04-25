
import scipy.stats
import matplotlib
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
from hydroDL.model import waterNetTest, waterNet
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
yP = outW['yP']

# LSTM
lstmOutName = '{}-{}'.format(dataName, trainSet)
yL, ycL = basinFull.testModel(
    lstmOutName, DF=DF, testSet=testSet, reTest=False, ep=epLSTM)
yL = yL[:, :, 0]

nash1 = utils.stat.calNash(yP, yT[nr-1:, :])
nash2 = utils.stat.calNash(yL[nr-1:, :], yT[nr-1:, :])
corr1 = utils.stat.calCorr(yP, yT[nr-1:, :])
corr2 = utils.stat.calCorr(yL[nr-1:, :], yT[nr-1:, :])
bias1 = utils.stat.calBiasR(yP, yT[nr-1:, :])
bias2 = utils.stat.calBiasR(yL, yT)
var1 = utils.stat.calVarR(yP, yT[nr-1:, :])
var2 = utils.stat.calVarR(yL, yT)
lat, lon = DF.getGeo()

importlib.reload(utils.stat)
importlib.reload(axplot)
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
dirFig = r'C:\Users\geofk\work\Presentation\AGU2022\posterFig'


fig, axes = figplot.boxPlot([[corr1**2, corr2**2], [corr1, corr2]],
                            label1=['NSE', 'Correlation'],
                            label2=['waterNet', 'LSTM'],
                            figsize=(8, 3))
fig.show()
fig.savefig(os.path.join(dirFig, 'box1.svg'))


fig, axes = figplot.boxPlot([[bias1, bias2], [var1, var2]],
                            label1=['Bias ratio', 'Variance ratio'],
                            yRange=[0.5, 1.5], figsize=(8, 3))
axes[0].axhline(1, color='k')
axes[1].axhline(1, color='k')
fig.show()
fig.savefig(os.path.join(dirFig, 'box2.svg'))


scipy.stats.wilcoxon(corr1, corr2)
scipy.stats.ttest_ind(corr1, corr2)


figM = plt.figure(figsize=(18, 6))
gsM = gridspec.GridSpec(1, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat, lon,
                        nash1-nash2, vRange=[-0.25, 0.25], s=35)
axM0.set_title('NSE(waterNet) - NSE(LSTM)')
figM.show()
figM.savefig(os.path.join(dirFig, 'diff.svg'))
