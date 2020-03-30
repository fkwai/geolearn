import statsmodels.api as sm
import os
import rnnSMAP
from rnnSMAP import runTrainLSTM, runTestLSTM
import matplotlib.pyplot as plt
import numpy as np
import imp
import time
import matplotlib
imp.reload(rnnSMAP)
rnnSMAP.reload()

doOpt = []
# doOpt.append('train')
# doOpt.append('test')
doOpt.append('loadData')
doOpt.append('plotConf')
doOpt.append('plotBar')

doOpt.append('plotBox')


rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma', 'eco_comb')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015, varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu',
        model='cudnn', loss='sigma'
    )
    for kk in range(1, 2):
        for k in range(11):
            trainName = 'ecoComb{}_v2f1'.format(k)
            opt['train'] = trainName
            if kk == 0:
                opt['var'] = 'varLst_soilM'
                opt['out'] = trainName+'_y15_soilM'
            elif kk == 1:
                opt['var'] = 'varLst_Forcing'
                opt['out'] = trainName+'_y15_Forcing'
            cudaID = k % 3
            print(trainName)
            runTrainLSTM.runCmdLine(
                opt=opt, cudaID=cudaID, screenName=opt['out'])

if 'test' in doOpt:
    for k in range(11):
        trainName = 'ecoComb{}_v2f1'.format(k)
        testName = 'ecoCombTest_v2f1'
        out = trainName+'_y15_Forcing'
        runTestLSTM.runCmdLine(
            rootDB=rootDB, rootOut=rootOut, out=out, testName=testName,
            yrLst=[2017], cudaID=k % 3, screenName=out)
        if k % 3 == 2:
            time.sleep(1000)


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
if 'plotConf' in doOpt:
    cLst = 'rygbbbbbbbb'
    strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
    titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    fig, axes = plt.subplots(ncols=len(strConfLst), figsize=(12, 4))
    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iHuc in range(11):
            temp = getattr(statConfLst[iHuc], strConfLst[k])
            plotLst.append(temp)
        if k < 2:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=None, cLst=cLst,
                xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
        else:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=None, cLst=cLst,
                xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
        print(out['ksdLst'])
        if k == 0:
            axes[k].set_ylabel('Frequency')
        axes[k].set_title(titleLst[k])
    saveFile = os.path.join(saveFolder, 'ecoComb_conf')
    # fig.show()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')

if 'plotBar' in doOpt:
    strConf='conf_sigma'
    labLst=['AB','CD','EF','ACE','ACF','ADE','ADF','BCE','BCF','BDE','BDF']
    figConf, axConf = plt.subplots(1,1, figsize=(12, 4))
    plotLst = list()
    for iHuc in range(11):
        temp = getattr(statConfLst[iHuc],strConf)
        plotLst.append(temp)
        _, _, out = rnnSMAP.funPost.plotCDF(plotLst,legendLst=None,showDiff=False)
    ksdLst=out['ksdLst']
    figBar, axBar= plt.subplots(1,1, figsize=(12, 4))
    axBar.bar(range(11), ksdLst,color='rrrbbbbbbbb')
    axBar.set_xticks(range(11))
    axBar.set_ylabel(r'd($p_{mc}$, 1-to-1)')
    axBar.set_xticklabels(labLst)
    saveFile = os.path.join(saveFolder, 'ecoComb_bar')
    # fig.show()
    figBar.savefig(saveFile, dpi=100)
    figBar.savefig(saveFile+'.eps')

#################################################
if 'plotBox' in doOpt:
    data = list()
    strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
    strErrLst = ['ubRMSE']
    labelC = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$',
              r'$\sigma_{mc} / \sigma_{x}$',
              'ubRMSE']
    for strSigma in strSigmaLst:
        temp = list()
        for k in range(11):
            statSigma = statSigmaLst[k]
            temp.append(getattr(statSigma, strSigma))
        data.append(temp)

    temp = list()
    for k in range(11):
        statSigma = statSigmaLst[k]
        rate = getattr(statSigma, 'sigmaMC')/getattr(statSigma, 'sigmaX')
        temp.append(rate)
    data.append(temp)

    for strErr in strErrLst:
        temp = list()
        for k in range(11):
            statErr = statErrLst[k]
            temp.append(getattr(statErr, strErr))
        data.append(temp)
    fig = rnnSMAP.funPost.plotBox(
        data, labelS=None, labelC=labelC, colorLst=cLst,
        figsize=(12, 4), sharey=False)
    fig.subplots_adjust(wspace=0.5)
    saveFile = os.path.join(saveFolder, 'ecoComb_box')
    # fig.show()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')


for k in range(11):
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    yLst = [statErrLst[k].ubRMSE, statErrLst[k].ubRMSE,
            statErrLst[k].ubRMSE, statSigmaLst[k].sigmaX]
    xLst = [statSigmaLst[k].sigmaX, statSigmaLst[k].sigmaMC,
            statSigmaLst[k].sigma, statSigmaLst[k].sigmaMC]
    xlabelLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$',
                 r'$\sigma_{comb}$', r'$\sigma_{x}$']
    ylabelLst = ['ubRMSE', 'ubRMSE', 'ubRMSE', r'$\sigma_{x}$']
    for i in range(4):
        rnnSMAP.funPost.plotVS(xLst[i], yLst[i], ax=axes[i],
                               xlabel=xlabelLst[i], ylabel=ylabelLst[i])
    saveFile = os.path.join(saveFolder, 'ecoComb{}_121'.format(k))
    fig.savefig(saveFile)


yLst = list()
x1Lst = list()
x2Lst = list()
x3Lst = list()
for k in range(11):
    yLst.append(statErrLst[k].ubRMSE)
    x1Lst.append(statSigmaLst[k].sigmaX)
    x2Lst.append(statSigmaLst[k].sigmaMC)
    x3Lst.append(statSigmaLst[k].sigma)
y = np.concatenate(yLst, axis=0)
x1 = np.concatenate(x1Lst, axis=0)
x2 = np.concatenate(x2Lst, axis=0)
x3 = np.concatenate(x3Lst, axis=0)
xx = np.stack((x1, x2), axis=1)
yy = y
ind = np.where(~np.isnan(yy))[0]
xf = xx[ind, :]
yf = yy[ind]
model = sm.RLM(yf, xf)
result = model.fit()
w = result.params
yp = result.predict(xx)


for k in range(11):
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    yLst = [statErrLst[k].ubRMSE, statErrLst[k].ubRMSE,
            statErrLst[k].ubRMSE, statErrLst[k].ubRMSE]
    xLst = [statSigmaLst[k].sigmaX, statSigmaLst[k].sigmaMC,
            statSigmaLst[k].sigma, statSigmaLst[k].sigmaX*w[0]+statSigmaLst[k].sigmaMC*w[1]]
    xlabelLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$',
                 r'$\sigma_{comb}$', r'$\sigma_{reg}$']
    ylabelLst = ['ubRMSE', 'ubRMSE', 'ubRMSE', r'$\sigma_{x}$']
    for i in range(4):
        rnnSMAP.funPost.plotVS(xLst[i], yLst[i], ax=axes[i],
                               xlabel=xlabelLst[i], ylabel=ylabelLst[i])
    saveFile = os.path.join(saveFolder, 'ecoComb{}_reg'.format(k))
    fig.savefig(saveFile)
