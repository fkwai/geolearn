import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imp
import statsmodels.api as sm
from rnnSMAP.funPost import distCDF
import matplotlib.gridspec as gridspec

imp.reload(rnnSMAP)
rnnSMAP.reload()

trainName = 'CONUSv2f1'
out = trainName + '_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma', 'regComb')

doOpt = []
doOpt.append('loadData')
doOpt.append('plotConf')
doOpt.append('plotMap')

optLst = [5, 6]
optEquLst = [
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2 + {:.2f} sigma_mc * sigma_mx',
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2',
]

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
plt.tight_layout()

figNameLst = ['Temporal', 'Spatial']
for figName in figNameLst:
#################################################
# load data
    if 'loadData' in doOpt:
        testSigmaLst = list()
        testConfLst = list()
        valSigmaLst = list()
        valConfLst = list()
        modelLst = list()
        if figName == 'Temporal':
            testName = 'CONUSv2f1'
            yr = [2017]
            valName = 'CONUSv2f1'
            valYr = [2016]
        elif figName == 'Spatial':
            testName = 'CONUSv2f2'
            yr = [2015]
            valName = 'CONUSv2fx2'
            valYr = [2015]

        ds = rnnSMAP.classDB.DatasetPost(rootDB=rootDB,
                                        subsetName=testName,
                                        yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

        dsVal = rnnSMAP.classDB.DatasetPost(rootDB=rootDB,
                                            subsetName=valName,
                                            yrLst=valYr)
        dsVal.readData(var='SMAP_AM', field='SMAP')
        dsVal.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

        testErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        valErr = dsVal.statCalError(predField='LSTM', targetField='SMAP')
        for (dsTemp, sigmaTempLst, confTempLst) in zip([ds, dsVal],
                                                    [testSigmaLst, valSigmaLst],
                                                    [testConfLst, valConfLst]):
            sigmaTemp = dsTemp.statCalSigma(field='LSTM')
            confTemp = dsTemp.statCalConf(predField='LSTM', targetField='SMAP')
            for sigmaStr in ['sigmaMC', 'sigmaX', 'sigma']:
                sigmaTempLst.append(getattr(sigmaTemp, sigmaStr + '_mat'))
                confTempLst.append(getattr(confTemp, 'conf_' + sigmaStr))
            for opt in optLst:
                print('doing option ' + str(opt))
                sigmaTemp, model = dsTemp.statRegSigma(dsVal, opt=opt)
                confTemp = dsTemp.statCalConf(predField='LSTM', targetField='SMAP')
                sigmaTempLst.append(sigmaTemp.sigmaReg_mat)
                confTempLst.append(confTemp.conf_sigmaReg)
                modelLst.append(model)

    #################################################
    # plot confidence figure
    if 'plotConf' in doOpt:
        figTitleLst = ['(a) {} Validation'.format(figName), '(b) {} Test'.format(figName)]
        fig, axes = plt.subplots(ncols=len(figTitleLst),
                                figsize=(12, 6),
                                sharey=True)
        sigmaStrLst = ['sigmaX', 'sigmaMC', 'sigma']
        legLst = [r'$p_{mc}$', r'$p_{x}$',  r'$p_{comb}$',r'$p_{reg1}$',r'$p_{reg2}$']
        for iFig in range(len(figTitleLst)):
            statConfLst = testConfLst if iFig == 1 else valConfLst
            rnnSMAP.funPost.plotCDF(
                statConfLst,
                ax=axes[iFig],
                legendLst=legLst,
                xlabel='Error Exceedance Probablity',
                ylabel=None,
                showDiff='KS')
            axes[iFig].set_title(figTitleLst[iFig])
        axes[0].set_ylabel('Frequency')
        # axes[1].get_legend().remove()
        fig.tight_layout()
        fig.show()
        saveFile = os.path.join(saveFolder, 'regComb_conf_'+figName)
        fig.savefig(saveFile)
        fig.savefig(saveFile+'.eps')

    # plot map
    if 'plotMap' in doOpt:
        optEquStrLst = [
            r'$\sigma_{{reg1}}={:.2f} \sigma_{{mc}}^2 + {:.2f} \sigma_x^2 + {:.2f} \sigma_{{mc}} \sigma_x$',
            r'$\sigma_{{reg2}}={:.2f} \sigma_{{mc}}^2 + {:.2f} \sigma_x^2$',
        ]
        fig = plt.figure(figsize=[12, 3])
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.5, 1, 0.5], height_ratios=[1])
        dataSigmaLst = [
            np.nanmean(testSigmaLst[-2], axis=1),
            np.nanmean(testSigmaLst[-1], axis=1)
        ]
        cRange = [0, 0.1]
        for k in range(2):
            ax = fig.add_subplot(gs[0, k * 2])
            grid = ds.data2grid(data=dataSigmaLst[k])
            wLst = modelLst[k].params.tolist()
            titleStr = optEquStrLst[k].format(*wLst)
            rnnSMAP.funPost.plotMap(grid,
                                    crd=ds.crdGrid,
                                    ax=ax,
                                    cRange=cRange,
                                    title=titleStr)
            ax = fig.add_subplot(gs[0, k * 2 + 1])
            ax.set_aspect('equal', 'box')
            rnnSMAP.funPost.plotVS(dataSigmaLst[k],
                                testErr.ubRMSE,
                                ax=ax,
                                xlabel=r'$\sigma_{{reg{}}}$'.format(k+1),
                                ylabel='ubRMSE')
            ax.set_xticks([0, 0.1])
            ax.set_yticks([0, 0.1])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.tight_layout()
        fig.show()
        saveFile = os.path.join(saveFolder, 'regComb_map_'+figName)
        fig.savefig(saveFile)
        fig.savefig(saveFile+'.eps')
