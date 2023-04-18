
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
# wnName = 'WaterNet0630'
epWN = 500
epLSTM = 500
modelFile = '{}-{}-ep{}'.format(wnName, dataName, epWN)
lstmOutName = '{}-{}'.format(dataName, trainSet)
dirOut = r'C:\Users\geofk\work\waterQuality\waterNet\outTemp'

# data
DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
ns = len(siteNoLst)

outName = 'gate{}'.format(modelFile)
outFile = os.path.join(dirOut, outName+'.npz')
outP = np.load(outFile)
lat, lon = DF.getGeo()
tabG = gageII.readData(siteNoLst=DF.siteNoLst)
tabG = gageII.updateCode(tabG)
dataG = tabG.values

['kp', 'ks', 'kg', 'gp', 'gL', 'qb', 'ga']
ga = outP['ga']
kp = outP['kp']
kg = outP['kg']
ks = outP['ks']
qb = outP['qb']*ga
par1 = np.nansum(ks*ga, axis=1)
par2 = np.nansum(kg*ga, axis=1)

dirFig = r'C:\Users\geofk\work\Presentation\AGU2022\posterFig'

figM = plt.figure(figsize=(15, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, par1, s=30)
figM.show()
figM.savefig(os.path.join(dirFig, 'k1.svg'))


figM = plt.figure(figsize=(15, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, par2, s=30)
figM.show()
figM.savefig(os.path.join(dirFig, 'k2.svg'))

figM = plt.figure(figsize=(15, 5))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, tabG['SLOPE_PCT'], s=30)
figM.show()
figM.savefig(os.path.join(dirFig, 'slope.svg'))
