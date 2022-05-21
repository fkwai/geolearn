
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
from hydroDL import utils
importlib.reload(waterNetTestC)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

# siteNo = '04193500'
siteNo = '09163500'
nm = 16

dataName = siteNo
DF = dbBasin.DataFrameBasin(dataName)

varX = ['pr', 'etr', 'tmmn', 'tmmx', 'srad', 'LAI']
mtdX = ['skip' for k in range(2)] +\
    ['scale' for k in range(2)] +\
    ['norm' for k in range(2)]
varY = ['runoff']+codeLst
mtdY = ['skip'] + ['scale' for code in codeLst]
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

# check data plot
labelLst = ['Q and P'] +\
    [usgs.codePdf.loc[code]['shortName'] for code in codeLst]
fig, axes = figplot.multiTS(DM0.t, DM0.y[:, 0, :], labelLst=labelLst)
ax = axes[0].twinx()
ax.plot(DM0.t, DM0.x[:, 0, 0], 'b')
ax.invert_yaxis()
fig.show()
# number of complete data
matNan = np.isnan(DM0.y[:, 0, 1:])
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
len(ind)

nh = 16
nr = 5
nc = len(codeLst)
model = waterNetTestC.Wn0119EMsolo(nh, nr, nc, nm)
saveDir = r'C:\Users\geofk\work\waterQuality\waterNet\modelTempEM'
modelFile = 'wn0119-{}-ep{}-nm{}'.format(dataName, 100, nm)
model.load_state_dict(torch.load(os.path.join(saveDir, modelFile)))


# test
model.eval()
figDir = os.path.join(
    r'C:\Users\geofk\work\waterQuality\waterNet\EMMA', siteNo)
if not os.path.exists(figDir):
    os.mkdir(figDir)
QLst = list()
for kk, (dataTup, t) in enumerate(zip([dataTup1, dataTup2], [DM1.t, DM2.t])):
    [x, xc, y, yc] = dataTup
    xP = torch.from_numpy(x).float().cuda()
    xcP = torch.from_numpy(xc).float().cuda()
    nt, ns, _ = y.shape
    yOut, qOut = model(xP, outQ=True)
    yP = yOut.detach().cpu().numpy()
    QpO, QsO, QgO = qOut
    Qa = yOut[:, 0, 0].detach().cpu().numpy()
    Qp = QpO.detach().cpu().numpy()[:, 0, :]
    Qs = QsO.detach().cpu().numpy()[:, 0, :]
    Qg = QgO.detach().cpu().numpy()[:, 0, :]
    QLst.append([Qa, Qp, Qs, Qg])
    corr = utils.stat.calCorr(yP[:, 0, :], y[nr-1:, 0, :])
    labelLst = list()
    for k in range(nc+1):
        if k == 0:
            labelLst.append('Q {:.2f}'.format(corr[k]))
        else:
            codeStr = usgs.codePdf.loc[codeLst[k-1]]['shortName']
            labelLst.append('{} {:.2f}'.format(codeStr, corr[k]))
    fig, axes = plt.subplots(nc+1, 1)
    fig, axes = figplot.multiTS(
        t[nr-1:], [yP[:, 0, :], y[nr-1:, 0, :]],
        labelLst=labelLst, cLst='rk', figsize=(8, 10))
    plt.tight_layout()
    fig.show()
    if kk == 0:
        fig.savefig(os.path.join(figDir, 'ts_train_nm{}'.format(nm)))
    if kk == 1:
        fig.savefig(os.path.join(figDir, 'ts_test_nm{}'.format(nm)))

# pars
cp = torch.relu(torch.exp(model.cp)-1
                ).repeat(int(nh/nm), 1).detach().cpu().numpy()
cs = torch.relu(torch.exp(model.cs)-1
                ).repeat(int(nh/nm), 1).detach().cpu().numpy()
cg = torch.relu(torch.exp(model.cg)-1
                ).repeat(int(nh/nm), 1).detach().cpu().numpy()
Qa, Qp, Qs, Qg = QLst[0]
Qcp1 = np.mean(Qp/Qa[:, None], axis=0)
Qcs1 = np.mean(Qp/Qa[:, None], axis=0)
Qcg1 = np.mean(Qp/Qa[:, None], axis=0)
Qa, Qp, Qs, Qg = QLst[1]
Qcp2 = np.mean(Qp/Qa[:, None], axis=0)
Qcs2 = np.mean(Qp/Qa[:, None], axis=0)
Qcg2 = np.mean(Qp/Qa[:, None], axis=0)


# pca and end members
pca = PCA(5)
[x, xc, y, yc] = dataTup0
temp = y[:, 0, 1:]
matNan = np.isnan(temp)
ind = np.where(np.sum(matNan, axis=-1) == 0)[0]
data = temp[ind, :]
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
    axes[k].set_ylabel('{:.1f}%'.format(r[1]*100))
    axes[k].legend()
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'pca_nm{}'.format(nm)))

# PCA unit
xu = np.eye(nc)
x0 = np.zeros([1, nc])
yu = pca.transform(xu)
y0 = pca.transform(x0)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
for k, code in enumerate(codeLst):
    ax.plot([y0[0, 0], yu[k, 0]], [y0[0, 1], yu[k, 1]], '-k')
    ax.text(yu[k, 0], yu[k, 1], usgs.codePdf.loc[code]['shortName'])
fig.show()


# # heat map of each EM
fig, axes = plt.subplots(nc+2, 1, figsize=(6, 10))
if nm < nh:
    n = int(nh/nm)
    ind = np.concatenate([np.arange(k, nh, n) for k in range(n)], axis=0)
else:
    ind = np.arange(nh)

for k, code in enumerate(codeLst):
    em = np.stack([cp[ind, k], cs[ind, k], cg[ind, k]])
    axplot.plotHeatMap(axes[k], em, fmt='{:.1f}')
    _ = axes[k].set_xticklabels([])
    _ = axes[k].set_yticklabels([])
    axes[k].set_ylabel(usgs.codePdf.loc[code]['shortName'])
for k, Q in enumerate(QLst):
    Qa, Qp, Qs, Qg = QLst[k]
    Qcp = np.mean(Qp/Qa[:, None], axis=0)*100
    Qcs = np.mean(Qs/Qa[:, None], axis=0)*100
    Qcg = np.mean(Qg/Qa[:, None], axis=0)*100
    em = np.stack([Qcp[ind], Qcs[ind], Qcg[ind]])
    axplot.plotHeatMap(axes[k+nc], em)
    _ = axes[k+nc].set_xticklabels([])
    _ = axes[k+nc].set_yticklabels([])
axes[nc].set_ylabel('Training Q [%]')
axes[nc+1].set_ylabel('Testing Q [%]')
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(figDir, 'em_nm{}'.format(nm)))

# Q compositation
Qa, Qp, Qs, Qg = QLst[1]
[x, xc, y, yc] = dataTup2
t = DM2.t
fig, axes = plt.subplots(4, 1)
axes[0].plot(t[nr-1:], Qa, 'r')
axes[0].plot(t, y[:, :, 0], 'k')
axes[1].plot(t[nr-1:], Qp)
axes[2].plot(t[nr-1:], Qs)
axes[3].plot(t[nr-1:], Qg)
fig.show()

# PCA of units
