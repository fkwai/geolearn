
import scipy
from sklearn.linear_model import LinearRegression
from hydroDL import utils
from sklearn.decomposition import PCA
import sklearn
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS

importlib.reload(waterNetTest)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']
# siteNo = '04193500'
siteNo = '09163500'
dataName = siteNo
DF = dbBasin.DataFrameBasin(dataName)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']
mtdY = ['skip']
varXC = gageII.varLstEx
mtdXC = ['skip' for var in varXC]
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)

# train
trainSet = 'WYB09'
testSet = 'WYA09'
DM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM1.trans(mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC)
dataTup1 = DM1.getData()
DM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM2.borrowStat(DM1)
dataTup2 = DM2.getData()
DM0 = dbBasin.DataModelBasin(
    DF, subset='all', varX=varX, varXC=varXC, varY=varY, varYC=varYC)
DM0.borrowStat(DM1)
dataTup0 = DM0.getData()

sizeLst = trainBasin.getSize(dataTup1)
[x, xc, y, yc] = dataTup1
[nx, nxc, ny, nyc, nt, ns] = sizeLst
batchSize = [1000, 100]
nh = 16
nr = 5
nc = len(codeLst)
model = waterNetTest.Wn0119solo(nh, nr)
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTempEM'
modelFile = 'wn0119-{}-ep{}'.format(dataName, 100)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))

# test
model.eval()
figDir = os.path.join(
    r'C:\Users\geofk\work\waterQuality\waterNet\EMMA', siteNo)
if not os.path.exists(figDir):
    os.mkdir(figDir)
varY = codeLst
mtdY = ['scale' for code in codeLst]
CM1 = dbBasin.DataModelBasin(
    DF, subset=trainSet, varX=None, varXC=None, varY=varY, varYC=None)
CM1.trans(mtdY=mtdY)
[_, _, c1, _] = CM1.getData()
CM2 = dbBasin.DataModelBasin(
    DF, subset=testSet, varX=None, varXC=None, varY=varY, varYC=None)
CM2.borrowStat(CM1)
[_, _, c2, _] = CM2.getData()
CM0 = dbBasin.DataModelBasin(
    DF, subset='all', varX=None, varXC=None, varY=varY, varYC=None)
CM0.borrowStat(CM1)
[_, _, c0, _] = CM0.getData()

# decomposition
QLst = list()
dataTup = dataTup1
[x, xc, y, yc] = dataTup
xP = torch.from_numpy(x).float().cuda()
yOut, qOut = model(xP, outQ=True)
QpO, QsO, QgO = qOut
Qa = yOut.detach().cpu().numpy()[:, 0]
Qp = QpO.detach().cpu().numpy()[:, 0, :]
Qs = QsO.detach().cpu().numpy()[:, 0, :]
Qg = QgO.detach().cpu().numpy()[:, 0, :]
QLst.append([Qa, Qp, Qs, Qg])

# reg training
cMat = np.zeros([nh*3, nc])
q1 = np.concatenate([Qp, Qs, Qg], axis=-1)/Qa[:, None]
# qAll = np.concatenate([Qp, Qs, Qg], axis=-1)/y[nr-1:, 0,:]
for k in range(nc):
    data = c1[nr-1:, 0, k]
    [c, q] = utils.rmNan([data, q1], returnInd=False)
    a, r = scipy.optimize.nnls(q, c)
    cMat[:, k] = a
    out = np.sum(q1*a, axis=1)
    utils.stat.calCorr(out, c1[nr-1:, 0, k])
cp1 = np.matmul(q1, cMat)
cp = cMat[:nh, :]
cs = cMat[nh:nh*2, :]
cg = cMat[nh*2:nh*3, :]

# training plot
labelLst = [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
pred = np.concatenate([Qa[:, None], cp1], axis=-1)
obs = np.concatenate([y[nr-1:, 0, :], c1[nr-1:, 0, :]], axis=-1)
corr = utils.stat.calCorr(obs, pred)
labelLst = list()
for k in range(nc+1):
    if k == 0:
        labelLst.append('Q {:.2f}'.format(corr[k]))
    else:
        codeStr = usgs.codePdf.loc[codeLst[k-1]]['shortName']
        labelLst.append('{} {:.2f}'.format(codeStr, corr[k]))
fig, axes = figplot.multiTS(DM1.t[nr-1:], [pred, obs], labelLst=labelLst,
                            cLst='rk', figsize=(8, 10))
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'ts_train_LSQ'))

# testing
dataTup = dataTup2
[x, xc, y, yc] = dataTup
xP = torch.from_numpy(x).float().cuda()
yOut, qOut = model(xP, outQ=True)
QpO, QsO, QgO = qOut
Qa = yOut.detach().cpu().numpy()[:, 0]
Qp = QpO.detach().cpu().numpy()[:, 0, :]
Qs = QsO.detach().cpu().numpy()[:, 0, :]
Qg = QgO.detach().cpu().numpy()[:, 0, :]
QLst.append([Qa, Qp, Qs, Qg])
q2 = np.concatenate([Qp, Qs, Qg], axis=-1)/Qa[:, None]
cp2 = np.matmul(q2, cMat)
# testing plot
pred = np.concatenate([Qa[:, None], cp2], axis=-1)
obs = np.concatenate([y[nr-1:, 0, :], c2[nr-1:, 0, :]], axis=-1)
corr = utils.stat.calCorr(obs, pred)
labelLst = list()
for k in range(nc+1):
    if k == 0:
        labelLst.append('Q {:.2f}'.format(corr[k]))
    else:
        codeStr = usgs.codePdf.loc[codeLst[k-1]]['shortName']
        labelLst.append('{} {:.2f}'.format(codeStr, corr[k]))
fig, axes = figplot.multiTS(DM2.t[nr-1:], [pred, obs], labelLst=labelLst,
                            cLst='rk', figsize=(8, 10))
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'ts_test_LSQ'))


# pca and end members
pca = PCA(5)
matNan = np.isnan(c0)
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
data = c0[ind, 0, :]
pca.fit(data)
r = pca.explained_variance_ratio_
out = pca.transform(data)
cpout = pca.transform(cp)
csout = pca.transform(cs)
cgout = pca.transform(cg)
fig, axes = plt.subplots(2, 1, figsize=(5, 10))
for k in range(2):
    axes[k].scatter(out[:, 0], out[:, 1], c='k', marker='*', s=5)
    Qa, Qp, Qs, Qg = QLst[k]
    Qcp = np.mean(Qp/Qa[:, None], axis=0)*100
    Qcs = np.mean(Qs/Qa[:, None], axis=0)*100
    Qcg = np.mean(Qg/Qa[:, None], axis=0)*100
    s = 15
    axes[k].scatter(cpout[:, 0], cpout[:, 1], s=Qcp*s,
                    facecolors='none', edgecolors='g', label='surface EM')
    axes[k].scatter(csout[:, 0], csout[:, 1], s=Qcs*s,
                    facecolors='none', edgecolors='r', label='shallow EM')
    axes[k].scatter(cgout[:, 0], cgout[:, 1], s=Qcg*s,
                    facecolors='none', edgecolors='b', label='deep EM')
    axes[k].set_xlabel('{:.1f}%'.format(r[0]*100))
    axes[k].set_xlabel('{:.1f}%'.format(r[1]*100))
    axes[k].legend()
plt.tight_layout()
fig.show()

fig, axes = plt.subplots(2, 1, figsize=(5, 10))
for k in range(2):
    axes[k].plot(out[:, 0], out[:, 1], 'k.')
    axes[k].plot(cpout[:, 0], cpout[:, 1], 'g*', label='surface EM')
    axes[k].plot(csout[:, 0], csout[:, 1], 'r*', label='shallow EM')
    axes[k].plot(cgout[:, 0], cgout[:, 1], 'b*', label='deep EM')
    axes[k].set_xlabel('{:.1f}%'.format(r[0]*100))
    axes[k].set_xlabel('{:.1f}%'.format(r[1]*100))
    axes[k].legend()
axes[1].set_xlim(-5, 5)
axes[1].set_ylim(-5, 5)
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'pca_LSQ'))

# # heat map of each EM
fig, axes = plt.subplots(nc+2, 1, figsize=(6, 10))
for k, code in enumerate(codeLst):
    em = np.stack([cp[:, k], cs[:, k], cg[:, k]])
    axplot.plotHeatMap(axes[k], em, fmt='{:.1f}')
    _ = axes[k].set_xticklabels([])
    _ = axes[k].set_yticklabels([])
    axes[k].set_ylabel(usgs.codePdf.loc[code]['shortName'])
for k, Q in enumerate(QLst):
    Qa, Qp, Qs, Qg = QLst[k]
    Qcp = np.mean(Qp/Qa[:, None], axis=0)*100
    Qcs = np.mean(Qs/Qa[:, None], axis=0)*100
    Qcg = np.mean(Qg/Qa[:, None], axis=0)*100
    em = np.stack([Qcp, Qcs, Qcg])
    axplot.plotHeatMap(axes[k+nc], em)
    _ = axes[k+nc].set_xticklabels([])
    _ = axes[k+nc].set_yticklabels([])
axes[nc].set_ylabel('Training Q [%]')
axes[nc+1].set_ylabel('Testing Q [%]')
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'em_LSQ'))
