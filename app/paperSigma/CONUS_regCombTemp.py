import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imp
import statsmodels.api as sm

imp.reload(rnnSMAP)
rnnSMAP.reload()

trainName = 'CONUSv2f1'
out = trainName+'_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

doOpt = []
doOpt.append('loadData')
doOpt.append('plotConf')
doOpt.append('plotCorr')
# doOpt.append('plotTemp')
# doOpt.append('plotBin')
# doOpt.append('plotProb')
optLst = [3, 4]
fTestLst = [[1, 1, 0, 0], [1, 1, 0]]

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
plt.tight_layout()

#################################################
# load data
if 'loadData' in doOpt:
    statSigmaLst = list()
    statConfLst = list()
    wLst = list()
    testName = 'CONUSv2f2'
    yr = [2015]
    valName = 'CONUSv2f1'
    valYr = [2016]

    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=yr)
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    dsVal = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=valName, yrLst=valYr)
    dsVal.readData(var='SMAP_AM', field='SMAP')
    dsVal.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    statErr = ds.statCalError(predField='LSTM', targetField='SMAP')

    for k in range(len(optLst)):
        statSigma = ds.statCalSigma(field='LSTM')
        w = statSigma.regComb(dsVal, opt=optLst[k], fTest=fTestLst[k])
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        wLst.append(w)

#################################################
# plot confidence figure
if 'plotConf' in doOpt:
    figTitleLst = ['option '+str(x) for x in optLst]
    fig, axes = plt.subplots(
        ncols=len(figTitleLst), figsize=(12, 6), sharey=True)
    sigmaStrLst = ['sigmaX', 'sigmaMC', 'sigma', 'sigmaReg']
    legendLst = [r'$p_{x}$', r'$p_{mc}$', r'$p_{comb}$', r'$p_{reg}$']
    for iFig in range(0, len(optLst)):
        statConf = statConfLst[iFig]
        figTitle = figTitleLst[iFig]
        plotLst = list()
        for k in range(0, len(sigmaStrLst)):
            plotLst.append(getattr(statConf, 'conf_'+sigmaStrLst[k]))

        _, _, out = rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[iFig], legendLst=legendLst, cLst='grbm',
            xlabel='Error Exceedance Probablity', ylabel=None, showDiff='KS')
        axes[iFig].set_title(figTitle)
        print(out['rmseLst'])
    axes[0].set_ylabel('Frequency')
    # axes[1].get_legend().remove()
    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'regComb_conf')
    fig.savefig(saveFile)
    # fig.savefig(saveFile+'.eps')


if 'plotCorr' in doOpt:
    figTitleLst = ['option '+str(x) for x in optLst]
    fig, axes = plt.subplots(
        ncols=len(figTitleLst), figsize=(12, 6))
    y = getattr(statErr, 'ubRMSE')
    for iFig in range(0, len(optLst)):
        x = getattr(statSigmaLst[iFig], 'sigmaX')
        ind = np.where(~np.isnan(x) & ~np.isnan(y))
        axes[iFig].set_aspect('equal', 'box')
        rnnSMAP.funPost.plotVS(x[ind], y[ind], ax=axes[iFig],
                               xlabel=r'$\sigma_{reg}$', ylabel='ubRMSE', title=figTitleLst[iFig])
    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'regComb_corr')
    # fig.savefig(saveFile)

#################################################
# plot confidence figure
if 'plotTemp' in doOpt:
    figTitleLst = [r'$\sigma_{mc}$ vs $\sigma_{x}$',
                   r'$a*\sigma_{mc}$ vs $\sigma_{true}$']
    fig, axes = plt.subplots(
        ncols=len(figTitleLst), figsize=(12, 6))

    sigmaTrue_mat = statSigma.sigmaX_mat-w[1]*statSigma.sigmaMC_mat
    sigmaTrue = np.mean(sigmaTrue_mat, axis=1)
    rnnSMAP.funPost.plotVS(
        statSigma.sigmaMC, statSigma.sigmaX, ax=axes[0], title=figTitleLst[0])
    rnnSMAP.funPost.plotVS(statSigma.sigmaMC*w[0], sigmaTrue,
                           ax=axes[1], title=figTitleLst[1])
    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'CONUS_regComb_corr_opt'+str(opt))
    # fig.savefig(saveFile, dpi=100)
    # fig.savefig(saveFile+'.eps')

# statSigma = dsVal.statCalSigma(field='LSTM')
# x1 = np.square(statSigma.sigmaMC_mat)
# x2 = np.square(statSigma.sigmaX_mat)
# x3 = statSigma.sigmaMC_mat*statSigma.sigmaX_mat
# x4 = np.ones(x1.shape)
# y = np.square(dsVal.LSTM-dsVal.SMAP)
# xx = np.stack((x1.flatten(), x2.flatten(),
#                x3.flatten(), x4.flatten()), axis=1)
# yy = y.flatten().reshape(-1, 1)

# ind = np.where(~np.isnan(yy))[0]
# xf = xx[ind, :]
# yf = yy[ind]
# w, _, _, _ = np.linalg.lstsq(xf, yf)
# model = sm.OLS(yf, xf)
# result = model.fit()
# ww = result.params
# ff=result.f_test([1, ww[1], ww[2], ww[3]])
# result.f_test([ww[0], 1, ww[2], ww[3]])
# result.f_test([ww[0], ww[1], 0, ww[3]])
# result.f_test([ww[0], ww[1], ww[2], 0])
# result.f_test(np.array([w,w]))
