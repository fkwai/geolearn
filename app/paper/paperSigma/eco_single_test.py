import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import huc_single_test
import imp
import matplotlib
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test huc vs huc

doOpt = []
doOpt.append('loadData')
# doOpt.append('crdMap')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')
# doOpt.append('plotConf')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

#################################################
# load data and plot map
for j in range(17):
    for k in range(17):
        trainName = 'ecoRegion'+str(j+1).zfill(2)+'_v2f1'
        testName = 'ecoRegion'+str(k+1).zfill(2)+'_v2f1'
        out = trainName+'_y15_Forcing'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(out=out, drMC=100, field='LSTM',
                    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'])

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
