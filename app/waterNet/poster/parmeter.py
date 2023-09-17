
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
# wnName = 'WaterNet0119'
wnName = 'WaterNet0630'
epWN = 500
epLSTM = 500
modelFile = '{}-{}-ep{}'.format(wnName, dataName, epWN)
lstmOutName = '{}-{}'.format(dataName, trainSet)
dirOut = r'C:\Users\geofk\work\waterQuality\waterNet\outTemp'

# data
# DF = dbBasin.DataFrameBasin(dataName)
siteNoLst = DF.siteNoLst
ns = len(siteNoLst)

outName = 'gate{}'.format(modelFile)
outFile = os.path.join(dirOut, outName+'.npz')
outP = np.load(outFile)
lat, lon = DF.getGeo()
tabG = gageII.readData(siteNoLst=DF.siteNoLst)
tabG = gageII.updateCode(tabG)
dataG = tabG.values
ns, ng = dataG.shape
tabG['ROCKDEPAVE']*tabG['AWCAVE']


['kp', 'ks', 'kg', 'gp', 'gL', 'qb', 'ga']
gL = outP['gL']
ga = outP['ga']
par = np.nansum(gL*ga, axis=1)

figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, par, s=20)
figM.show()

rMat = np.ndarray(ng)
for k in range(ng):
    rMat[k] = np.corrcoef(par, dataG[:, k])[0, 1]

ind = np.argsort(np.abs(rMat))
nNan = np.sum(np.isnan(rMat))

k = 1
rMat[ind[-nNan-k]]
tabG.columns[ind[-nNan-k]]

# par = np.nanmean(gL, axis=1)
a = par
b = tabG['AWCAVE']
np.corrcoef(a, b)

lat, lon = DF.getGeo()
figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, par, s=20)
figM.show()


figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, tabG['AWCAVE'], s=20)
figM.show()

figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat,
                       lon, tabG['ROCKDEPAVE']*tabG['BDAVE'], s=20)
figM.show()


['kp', 'ks', 'kg', 'gp', 'gL', 'qb', 'ga']
kp = outP['kp']
kg = outP['kg']
ks = outP['ks']
qb = outP['qb']*ga
par = np.nansum(ks*ga, axis=1)
par = np.nansum(qb/kg*ga, axis=1)


figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, par, s=20)
figM.show()

figM = plt.figure()
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, tabG['SLOPE_PCT'], s=20)
figM.show()

a = par
b = tabG['PERMAVE']
np.corrcoef(a, b)

fig, ax = plt.subplots(1, 1)
ax.plot(a, b, '*')
fig.show()
