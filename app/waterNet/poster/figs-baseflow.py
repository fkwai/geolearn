
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
ga = outP['ga']
qb = outP['qb']
kg = outP['kg']

tabG = gageII.readData(siteNoLst=DF.siteNoLst)
tabG = gageII.updateCode(tabG)


bi = np.sum(QgP, axis=(0, 2))/np.sum(yP, axis=0)

np.sum(QgP+QsP+QpP, axis=2)-yP

lat, lon = DF.getGeo()

figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM0 = mapplot.mapPoint(figM, gsM[0, 0], lat,
                        lon, bi*100, vRange=[0, 100], s=20)
figM.show()


figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat,
                       lon, tabG['BFI_AVE'],  vRange=[0, 100], s=20)
figM.show()

# total storage
aa = np.nanmean(np.nansum(SgP*ga, axis=2))+np.nansum(qb/kg*ga, axis=1)
figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat,lon, aa, s=20)
figM.show()