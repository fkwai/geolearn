import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

ecoLst = ['AB', 'CD', 'EF', 'ACE', 'ACF',
          'ADE', 'ADF', 'BCE', 'BCF', 'BDE', 'BDF']

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma', 'eco_comb')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

doOpt = []
doOpt.append('loadData')
doOpt.append('plotBin')
# doOpt.append('plotProb')

#################################################
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(11):
        trainName = 'ecoComb{}_v2f1'.format(k)
        out = trainName+'_y15_Forcing'
        testName = 'ecoCombTest_v2f1'

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statErrLst.append(statErr)
        statSigma = ds.statCalSigma(field='LSTM')
        statSigmaLst.append(statSigma)
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statConfLst.append(statConf)


#################################################
# plot confidence figure
if 'plotBin' in doOpt:
    cLst = [[1, 0.5, 0.5],
            [1, 0.5, 0.0],
            [1, 0, 0.5],
            [0., 0., 1],
            [0., 0.3, 1],
            [0., 0.8, 1],
            [0., 1, 1],
            [0., 1, 0.8],
            [0., 1, 0.3],
            [0., 1, 0],
            [0., 0.5, 0.5]]
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for iEco in range(0, len(ecoLst)):
        sigmaMC = getattr(statSigmaLst[iEco], 'sigmaMC_mat')
        sigmaX = getattr(statSigmaLst[iEco], 'sigmaX_mat')
        sigma = getattr(statSigmaLst[iEco], 'sigma')
        dataBin = sigmaMC/sigmaX
        # dataBin = sigma
        ubRMSE = getattr(statErrLst[iEco], 'ubRMSE')
        confMat = getattr(statConfLst[iEco], 'conf_sigma')
        nbin = 10
        xbin = np.percentile(dataBin, range(0, 101, int(100/nbin)))
        xbinMean = (xbin[0:nbin]+xbin[1:nbin+1])/2
        corrLst = list()
        distLst = list()
        for k in range(0, nbin):
            ind = (dataBin > xbin[k]) & (dataBin <= xbin[k+1])
            conf = rnnSMAP.funPost.flatData(confMat[ind])
            if k == 0:
                print(iEco, len(conf))
            yRank = np.arange(len(conf))/float(len(conf)-1)
            dist = np.abs(conf - yRank).max()
            # dist=np.nanmax(conf-yRank)
            distLst.append(dist)
        ax.plot(xbinMean, distLst, marker='*',
                color=cLst[iEco], label=ecoLst[iEco])
    ax.set_ylabel(r'd($p_{mc}$, 1-to-1)')
    ax.set_xlabel(r'$\sigma_{mc}$ / $\sigma_{x}$')
    # ax.set_xlabel(r'$\sigma_{x}$')
    ax.legend()
    fig.show()
    # saveFile = os.path.join(saveFolder, 'CONUS_sigmaRatioBin')
    saveFile = os.path.join(saveFolder, 'sigma_bin')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
