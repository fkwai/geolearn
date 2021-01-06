import time
import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imp
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
# doOpt.append('plotBin')
# doOpt.append('plotProb')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
plt.tight_layout()

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    statProbLst = list()

    for k in range(0, 2):
        if k == 0:  # validation
            testName = 'CONUSv2f1'
            yr = [2016]
        if k == 1:  # temporal test
            testName = 'CONUSv2f1'
            yr = [2017]

        predField = 'LSTM'
        targetField = 'SMAP'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statProb = ds.statCalProb(predField='LSTM', targetField='SMAP')
        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        statProbLst.append(statProb)

# do regression
x1 = statSigmaLst[0].sigmaX_mat
x2 = statSigmaLst[0].sigmaMC_mat
x3 = np.multiply(x1, x2)
y = dsLst[0].LSTM-dsLst[0].SMAP
xx = np.stack((x1.flatten(), x2.flatten(), x3.flatten()), axis=1)
yy =np.abs(y.flatten()).reshape(-1, 1)
ind = np.where(~np.isnan(yy))[0]
xf = xx[ind, :]
yf = yy[ind]

w,res,rank,s=np.linalg.lstsq(xf,yf)

ss = statSigmaLst[1]
a = ss.sigmaX_mat*w[0]+ss.sigmaMC_mat*w[1]+ss.sigmaX_mat*ss.sigmaMC_mat*w[2]
# import time
# t0 = time.time()
# out = rnnSMAP.funPost.regLinear(yy, xx)
# tt = time.time()-t0
# print(tt)
