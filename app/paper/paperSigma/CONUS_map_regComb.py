import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

figTitleLst = ['Temporal Test', 'Spatial Test']
figNameLst = ['temporal_reg', 'spatial_reg']

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})


for iFig in range(0, 2):
    # iFig = 0
    figTitle = figTitleLst[iFig]
    if iFig == 0:
        testName = 'CONUSv2f1'
        yr = [2017]
        valName = 'CONUSv2fy2'
        valYr = [2015]
    if iFig == 1:
        testName = 'CONUSv2fx2'
        yr = [2015]
        valName = 'CONUSv2fy2'
        valYr = [2015]

    trainName = 'CONUSv2f1'
    out = trainName+'_y15_Forcing_dr60'
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    caseStrLst = ['sigmaMC', 'sigmaX', 'sigma']
    nCase = len(caseStrLst)
    saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

    #################################################
    # test
    predField = 'LSTM'
    targetField = 'SMAP'

    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=yr)
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    dsVal = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=valName, yrLst=valYr)
    dsVal.readData(var='SMAP_AM', field='SMAP')
    dsVal.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds.statCalSigma(field='LSTM')
    statSigma.regComb(dsVal)
    statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
    statNorm = rnnSMAP.classPost.statNorm(
        statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

    #################################################
    # plot figure
    fig = plt.figure(figsize=[12, 3])
    gs = gridspec.GridSpec(
        1, 3, width_ratios=[1, 1, 0.5], height_ratios=[1])

    dataErr = getattr(statErr, 'ubRMSE')
    dataSigma = getattr(statSigma, 'sigmaReg')
    cRange = [0, 0.1]

    # plot map RMSE
    ax = fig.add_subplot(gs[0, 0])
    grid = ds.data2grid(data=dataErr)
    titleStr = 'ubRMSE of '+figTitle
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                            cRange=cRange, title=titleStr)
    # plot map sigma
    ax = fig.add_subplot(gs[0, 1])
    grid = ds.data2grid(data=dataSigma)
    titleStr = r'$\sigma_{reg}$'+' of '+figTitle
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                            cRange=cRange, title=titleStr)
    fig.show()
    # plot map sigma vs RMSE
    ax = fig.add_subplot(gs[0, 2])
    ax.set_aspect('equal', 'box')
    y = dataErr
    x = dataSigma
    rnnSMAP.funPost.plotVS(
        x, y, ax=ax, xlabel=r'$\sigma_{reg}$', ylabel='ubRMSE')

    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'map_'+figNameLst[iFig])
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')

    #################################################
    # plot sigmaX vs sigmaMC
    plotSigma = 1
    if plotSigma == 1:
        fig = plt.figure(figsize=[12, 3])
        gs = gridspec.GridSpec(
            1, 3, width_ratios=[1, 1, 0.5], height_ratios=[1])

        dataSigmaX = getattr(statSigma, 'sigma')
        dataSigmaMC = getattr(statSigma, 'sigmaReg')

        # plot map RMSE
        ax = fig.add_subplot(gs[0, 0])
        grid = ds.data2grid(data=dataSigmaX)
        titleStr = r'$\sigma_{comb}$ '+figTitle
        rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                                cRange=[0, 0.1], title=titleStr)

        # plot map sigma
        ax = fig.add_subplot(gs[0, 1])
        grid = ds.data2grid(data=dataSigmaMC)
        titleStr = r'$\sigma_{reg}$'+' of '+figTitle
        rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                                cRange=[0, 0.1], title=titleStr)

        # plot map sigma vs RMSE
        ax = fig.add_subplot(gs[0, 2])
        ax.set_aspect('equal', 'box')
        y = dataSigmaMC
        x = dataSigmaX
        rnnSMAP.funPost.plotVS(
            x, y, ax=ax, xlabel=r'$\sigma_{comb}$', ylabel=r'$\sigma_{reg}$')

        fig.tight_layout()
        fig.show()
        saveFile = os.path.join(saveFolder, 'map_'+figNameLst[iFig]+'_sigma')
        fig.savefig(saveFile, dpi=100)
        fig.savefig(saveFile+'.eps')
