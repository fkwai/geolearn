
# %%
import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from rnnSMAP import runTestLSTM
import shapefile
import time
import imp
import math
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# train on one HUC and test on CONUS. look at map of sigma

doOpt = []
# doOpt.append('test')
doOpt.append('loadData')
# doOpt.append('plotMapMC')
# doOpt.append('plotMapPaper')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma', 'eco_single_map')
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['Bias', 'ubRMSE']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
yrLst = [2017]

hucShapeFile = '/mnt/sdb/Kuai/map/ecoRegion/ecoRegion'
shapeLst = shapefile.Reader(hucShapeFile).shapes()
shapeRecLst = shapefile.Reader(hucShapeFile).records()
ecoIdLst = [rec[1] for rec in shapeRecLst]

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})

# %% load data
dsLst = list()
statErrLst = list()
statSigmaLst = list()
# for k in range(17):
k = 5
ecoId = k+1
trainName = 'ecoRegion'+str(ecoId).zfill(2)+'_v2f1'
testName = 'CONUSv2f1'
out = trainName+'_y15_Forcing'
ds = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=yrLst)
ds.readData(var='SMAP_AM', field='SMAP')
ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
dsLst.append(ds)

statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
statSigma = ds.statCalSigma(field='LSTM')
statErrLst.append(statErr)
statSigmaLst.append(statSigma)

#################################################
# %% plot map
imp.reload(rnnSMAP)
rnnSMAP.reload()
k = 0
ecoId = 6
shape = shapeLst[ecoIdLst.index(ecoId)]
statSigma = statSigmaLst[k]
statErr = 'sigmaMC'
fig, ax = plt.subplots(figsize=[6, 3])
data = statSigma.sigmaMC
grid = ds.data2grid(data=data)
titleStr = r'$\sigma_{mc}$' + ' from Eco-region %02d model' % (ecoId)
rnnSMAP.funPost.plotMap(
    grid, crd=ds.crdGrid, ax=ax, title=titleStr,
    shape=shape)
plt.tight_layout()
fig.show()
saveFile = os.path.join(saveFolder, 'map_sigmaMC_%02d' % (ecoId))
# fig.savefig(saveFile)


# %%
