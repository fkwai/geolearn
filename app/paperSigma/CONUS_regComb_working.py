import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imp
import statsmodels.api as sm
from rnnSMAP import funPost

imp.reload(rnnSMAP)
rnnSMAP.reload()

trainName = 'CONUSv2f1'
out = trainName + '_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma', 'regComb')

doOpt = []
doOpt.append('loadData')
doOpt.append('print')
# doOpt.append('plotConf')
# doOpt.append('doTest')
# doOpt.append('plotCorr')
# doOpt.append('plotTemp')
# doOpt.append('plotBin')
# doOpt.append('plotProb')
optLst = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# optLst = [5, 6]

optEquLst = [
    '{:.2f} sigma_mc^2 + {:.2f} sigma_mc * sigma_x^2 + sigma_x^2',
    '{:.2f} sigma_mc^2 + sigma_x^2',
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2 + {:.2f} sigma_mc * sigma_x + {:.5f}',
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2 + {:.5f}',
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2 + {:.2f} sigma_mc * sigma_x',
    '{:.2f} sigma_mc^2 + {:.2f} sigma_x^2', '{:.2f} sigma_mc^2',
    '{:.2f} sigma_x^2', 'sigma_mc^2 + sigma_x^2 + {:.8f}'
]
fTestLst = [[1, 1, 0, 0], [1, 1, 0]]

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})
plt.tight_layout()

#################################################
# load data
if 'loadData' in doOpt:
    testSigmaLst = list()
    testConfLst = list()
    valSigmaLst = list()
    valConfLst = list()
    modelLst = list()
    testName = 'CONUSv2f2'
    yr = [2015]
    valName = 'CONUSv2fy2'
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
            sigmaTemp, model = dsTemp.statRegSigma2(dsVal, opt=opt)
            confTemp = dsTemp.statCalConf(predField='LSTM', targetField='SMAP')
            sigmaTempLst.append(sigmaTemp.sigmaReg_mat)
            confTempLst.append(confTemp.conf_sigmaReg)
            modelLst.append(model)

if 'print' in doOpt:
    labelLst = ['sigmaMC', 'sigmaX', 'sigmaComb'] +\
        ['sigmaReg opt '+str(x) for x in optLst]

    # calculate cdf distance
    testRmseLst, testKsdLst = funPost.distCDF(testConfLst)
    valRmseLst, valKsdLst = funPost.distCDF(valConfLst)

    print('# Regressed equation')
    for k in range(len(optLst)):
        wLst = modelLst[k].params.tolist()
        print('opt {}: '.format(optLst[k]) + optEquLst[k].format(*wLst))

    # residual
    valSsrLst = list()
    testSsrLst = list()
    for (ssrLst, sigmaLst, dsTemp) in zip([testSsrLst, valSsrLst],
                                        [testSigmaLst, valSigmaLst],
                                        [ds, dsVal]):
        for (sigma, label) in zip(sigmaLst, labelLst):
            res = np.square(dsTemp.LSTM - dsTemp.SMAP) - np.square(sigma)
            ssr = np.nansum(np.square(res.flatten()))
            ssrLst.append(ssr)

    # corr
    valCorrLst = list()
    testCorrLst = list()
    for (corrLst, sigmaLst, err) in zip([testCorrLst, valCorrLst],
                                        [testSigmaLst, valSigmaLst],
                                        [testErr, valErr]):
        for (sigma, label) in zip(sigmaLst, labelLst):
            ubRMSE = err.ubRMSE
            corr = np.corrcoef(ubRMSE, np.nanmean(sigma, axis=1))[0, 1]
            corrLst.append(corr)
            # print('{}: R(ubRMSE,sigma) = {}'.format(label, corr))

    print('# Validation set')
    for k in range(len(valSigmaLst)):
        print('validation {}: R = {:.4f}, KS = {:.4f}'.format(
            labelLst[k], valCorrLst[k], valKsdLst[k]))
    print('# testing set')
    for k in range(len(testSigmaLst)):
        print('test {}: R = {:.4f}, KS = {:.4f}'.format(labelLst[k],
                                                        testCorrLst[k],
                                                        testKsdLst[k]))

#################################################
# plot confidence figure
if 'plotConf' in doOpt:
    figTitleLst = ['Validation yr' + str(valYr[0]), 'Test yr' + str(yr[0])]

    fig, axes = plt.subplots(ncols=len(figTitleLst),
                             figsize=(12, 6),
                             sharey=True)
    sigmaStrLst = ['sigmaX', 'sigmaMC', 'sigma']
    legLst = [r'$p_{mc}$', r'$p_{x}$',  r'$p_{comb}$'] +\
        [r'$p_{reg}$ opt '+str(x) for x in optLst]
    for iFig in range(len(figTitleLst)):
        statConfLst = testConfLst if iFig == 1 else valConfLst
        _, _, out = rnnSMAP.funPost.plotCDF(
            statConfLst,
            ax=axes[iFig],
            legendLst=legLst,
            xlabel='Error Exceedance Probablity',
            ylabel=None,
            showDiff='KS')
        axes[iFig].set_title(figTitleLst[iFig])
        print(out['rmseLst'])
    axes[0].set_ylabel('Frequency')
    # axes[1].get_legend().remove()
    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'regComb_conf_reg')
    fig.savefig(saveFile)
    # fig.savefig(saveFile+'.eps')

if 'doTest' in doOpt:

    # import statsmodels.api as sm
    testName = 'CONUSv2f1'
    yr = [2016]
    valName = 'CONUSv2f1'
    valYr = [2017]

    trainName = 'CONUSv2f1'
    out = trainName + '_y15_Forcing_dr60'
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

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

    dsReg = dsVal
    predField = 'LSTM'
    targetField = 'SMAP'
    statSigma = dsReg.statCalSigma(field=predField)
    statErr = ds.statCalError(predField=predField, targetField=targetField)
    y = statErr.ubRMSE

    x1 = np.square(statSigma.sigmaMC)
    x2 = statSigma.sigmaMC * statSigma.sigmaX
    xx = np.stack((x1, x2), axis=1)
    yy = y - np.square(statSigma.sigmaX)

    ind = np.where(~np.isnan(yy))[0]
    xf = xx[ind, :]
    yf = yy[ind]
    model = sm.OLS(yf, xf)
    result = model.fit()
    yp = result.predict(xf).flatten().astype(np.float32)
    yf = yf.flatten().astype(np.float32)
