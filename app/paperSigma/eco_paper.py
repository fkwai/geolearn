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
doOpt.append('plotMapPaper')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma', 'eco_single_map')
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['Bias', 'ubRMSE']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
yrLst = [2017]

# ecoShapeFile = r'C:\Users\geofk\work\map\ecoRegion\comb\ecoRegion'
ecoShapeFile = '/mnt/sdb/Kuai/map/ecoRegion/ecoRegion'
shapeLst = shapefile.Reader(ecoShapeFile).shapes()
shapeRecLst = shapefile.Reader(ecoShapeFile).records()
ecoIdLst = [rec[1] for rec in shapeRecLst]

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})

codeLst = ['8.3', '9.4', '10.1', '10.2']
idLst = [5, 10, 12, 13]
#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for ecoId in idLst:
        print('load {}'.format(ecoId))
        k = ecoId-1
        trainName = 'ecoRegion'+str(k+1).zfill(2)+'_v2f1'
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
if 'plotMapPaper' in doOpt:
    figNum = ['(a)', '(b)', '(c)', '(d)']
    print('111')
    fig, axes = plt.subplots(2, 2, figsize=[12, 7])
    print('222')
    for a, ecoId in enumerate(idLst):
        k = ecoId-1
        shape = shapeLst[ecoIdLst.index(ecoId)]
        statSigma = statSigmaLst[a]
        statErr = 'sigmaMC'
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = figNum[a]+' ' + \
            r'$\sigma_{mc}$' + ' from Eco-region {} model'.format(codeLst[k])
        ax = axes[math.floor(a/2), a % 2]
        print('1 plot')
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shape)
    print('2 plot')
    plt.tight_layout()
    # fig.show()
    saveFile = os.path.join(saveFolder, 'map_sigmaMC')
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
    print(saveFile)
