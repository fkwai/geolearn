import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import pylab
import scipy.stats as stats
import matplotlib.gridspec as gridspec

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma', 'invG')

trainName = 'CONUSv4f1'
testName = 'CONUSv4f1'
yrLst = [2015, 2016, 2017]

for yr in yrLst:
    outLst = ['CONUSv4f1_y15_Forcing_dr60']
    figNameLst = ['_NoPrior']
    figTitleLst = ['Temporal Test without prior']
    # C1Lst = [4, 2]
    # C2Lst = [2, 1]
    # for C1, C2 in zip(C1Lst, C2Lst):
    for C1 in C1Lst:
        for C2 in C2Lst:
            C1Lst = [2, 3, 4]
            C2Lst = [1, 2, 4]
            out = 'CONUSv4f1_y15_Forcing_dr60_invGamma_{}_{}'.format(C1, C2)
            figName = '_{}_{}'.format(C1, C2)
            figTitle = 'Temporal Test with prior invGamma({},{})'.format(
                C1-1, C2/2)
            outLst.append(out)
            figNameLst.append(figName)
            figTitleLst.append(figTitle)

    for out, figName, figTitle in zip(outLst, figNameLst, figTitleLst):
        predField = 'LSTM'
        targetField = 'SMAP'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[yr])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statNorm = rnnSMAP.classPost.statNorm(
            statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

        # plot figure
        fig = plt.figure(figsize=[12, 3])
        gs = gridspec.GridSpec(
            1, 3, width_ratios=[1, 1, 0.5], height_ratios=[1])

        dataErr = getattr(statErr, 'RMSE')
        dataSigma = getattr(statSigma, 'sigma')
        cRange = [0, 0.1]

        # plot map RMSE
        ax = fig.add_subplot(gs[0, 0])
        grid = ds.data2grid(data=dataErr)
        titleStr = 'RMSE'
        rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                                cRange=cRange, title=titleStr)
        # plot map sigma
        ax = fig.add_subplot(gs[0, 1])
        grid = ds.data2grid(data=dataSigma)
        titleStr = r'$\sigma_{comb}$'
        rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                                cRange=cRange, title=titleStr)
        fig.show()
        # plot map sigma vs RMSE
        ax = fig.add_subplot(gs[0, 2])
        ax.set_aspect('equal', 'box')
        y = dataErr
        x = dataSigma
        rnnSMAP.funPost.plotVS(
            x, y, ax=ax, xlabel=r'$\sigma_{comb}$', ylabel='RMSE')

        fig.suptitle(figTitle)
        fig.tight_layout()
        fig.show()
        saveFile = os.path.join(saveFolder, 'map{}_{}_'.format(figName,yr))
        fig.savefig(saveFile, dpi=100)
        fig.savefig(saveFile+'.eps')
