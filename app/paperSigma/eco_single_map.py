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

codeLst = ['5.1', '6.2', '8.1', '8.2', '8.3', '8.4', '8.5', '9.2',
           '9.3', '9.4', '9.5', '10.1', '10.2', '11.1', '12.1', '13.1', '14.3']
idLst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
nameLst = ['Northern Forests', 'Northwestern Forests Mountain', 'Mixed Wood Plains', 'Central USA Plains', 'Southeastern USA Plains', 'Ozark/Ouachita-Appalachiana Forests', 'Mississippi Alluvial and Southeast Coastal',
           'Temperate Prairies', 'West-central Semiarid Prairies', 'South-central Semiarid Prairies', 'Texas Plain', 'Cold Deserts', 'Warm Deserts', 'Mediterranean California', 'Southern Semiarid Highlands', 'Temperate Sierras', 'Tropical Forests']


#################################################
# test
if 'test' in doOpt:
    for k in range(0, 17):
        trainName = 'ecoRegion'+str(k+1).zfill(2)+'_v2f1'
        testName = 'CONUSv2f1'
        out = trainName+'_y15_Forcing'
        runTestLSTM.runCmdLine(
            rootDB=rootDB, rootOut=rootOut, out=out, testName=testName,
            yrLst=yrLst, cudaID=k % 3, screenName=out)
        # if k % 3 == 2:
        # time.sleep(1000)

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(17):
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
if 'plotMapMC' in doOpt:
    for k in range(17):
        # for k in [5]:
        ecoId = k+1
        shape = shapeLst[ecoIdLst.index(ecoId)]
        statSigma = statSigmaLst[k]
        statErr = 'sigmaMC'
        fig, ax = plt.subplots(figsize=[6, 3])
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = r'$\sigma_{mc}$' + ' from Eco-region %02d model' % (k+1)
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shape)
        plt.tight_layout()
        # fig.show()
        saveFile = os.path.join(saveFolder, 'map_sigmaMC_%02d' % (k+1))
        fig.savefig(saveFile)

#################################################
if 'plotMapPaper' in doOpt:
    a = 0
    figNum = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(2, 2, figsize=[12, 7])
    for ecoId in [5, 10, 12, 13]:
        k = ecoId-1
        shape = shapeLst[ecoIdLst.index(ecoId)]
        statSigma = statSigmaLst[k]
        statErr = 'sigmaMC'
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = figNum[a]+' ' + \
            r'$\sigma_{mc}$' + ' from Eco-region {} model'.format(codeLst[k])
        ax = axes[math.floor(a/2), a % 2]
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shape)
        a = a+1
    plt.tight_layout()
    # fig.show()
    saveFile = os.path.join(saveFolder, 'map_sigmaMC')
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
    print(saveFile)
