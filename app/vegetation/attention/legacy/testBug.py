import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os
from hydroDL import utils
from hydroDL.post import mapplot, axplot, figplot
import matplotlib.pyplot as plt
from hydroDL.model import rnn, crit, trainBasin
import math
import torch
from torch import nn
from hydroDL.data import DataModel
from hydroDL.master import basinFull, slurm
import torch.optim as optim
from hydroDL import kPath
import torch.optim.lr_scheduler as lr_scheduler
import dill


def dataTs2Range(dataTup, rho, returnInd=False):
    # assuming yc is none
    x, xc, y, yc = dataTup
    nt = y.shape[0]
    jL, iL = np.where(~np.isnan(y).any(axis=-1))
    xLst, xcLst, ycLst = list(), list(), list()
    jLout, iLout = list(), list()
    for j, i in zip(jL, iL):
        if j >= rho and j < nt - rho:
            if x is not None:
                xLst.append(x[j - rho : j + rho + 1, i, :])
            if xc is not None:
                xcLst.append(xc[i, :])
            if yc is None:
                ycLst.append(y[j, i, :])
            iLout.append(i)
            jLout.append(j)
    xE = np.stack(xLst, axis=0)
    xE = xE.swapaxes(0, 1)
    xcE = np.stack(xcLst, axis=0)
    ycE = np.stack(ycLst, axis=0)
    if returnInd:
        return (xE, xcE, None, ycE), (jLout, iLout)
        # return (xE, xcE, None, ycE), (jL, iL)
    else:
        return (xE, xcE, None, ycE)


def dataTs2Range_bug(dataTup, rho, returnInd=False):
    # assuming yc is none
    x, xc, y, yc = dataTup
    nt = y.shape[0]
    jL, iL = np.where(~np.isnan(y).any(axis=-1))
    xLst, xcLst, ycLst = list(), list(), list()
    jLout, iLout = list(), list()
    for j, i in zip(jL, iL):
        if j >= rho and j < nt - rho:
            if x is not None:
                xLst.append(x[j - rho : j + rho + 1, i, :])
            if xc is not None:
                xcLst.append(xc[i, :])
            if yc is None:
                ycLst.append(y[j, i, :])
            iLout.append(i)
            jLout.append(j)
    xE = np.stack(xLst, axis=0)
    xE = xE.swapaxes(0, 1)
    xcE = np.stack(xcLst, axis=0)
    ycE = np.stack(ycLst, axis=0)
    if returnInd:
        # return (xE, xcE, None, ycE), (jLout, iLout)
        return (xE, xcE, None, ycE), (jL, iL)
    else:
        return (xE, xcE, None, ycE)


rho = 45
dataName = "singleDaily"
importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameVeg(dataName)
dm = DataModel(X=df.x, XC=df.xc, Y=df.y)
siteIdLst = df.siteIdLst
dm.trans(mtdDefault="minmax")
dataTup = dm.getData()
dataEnd, (iInd2, jInd2) = dataTs2Range_bug(dataTup, rho, returnInd=True)
dataEnd, (iInd, jInd) = dataTs2Range(dataTup, rho, returnInd=True)


x, xc, y, yc = dataEnd

np.nanmean(dm.x[:, :, 0])
np.nanmax(df.x[:, :, 2])

# calculate position
varS = ["VV", "VH", "vh_vv"]
varL = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "ndvi", "ndwi", "nirv"]
varM = ["Fpar", "Lai"]
iS = [df.varX.index(var) for var in varS]
iL = [df.varX.index(var) for var in varL]
iM = [df.varX.index(var) for var in varM]

pSLst, pLLst, pMLst = list(), list(), list()
ns = yc.shape[0]
nMat = np.zeros([yc.shape[0], 3])
for k in range(nMat.shape[0]):
    tempS = x[:, k, iS]
    pS = np.where(~np.isnan(tempS).any(axis=1))[0]
    tempL = x[:, k, iL]
    pL = np.where(~np.isnan(tempL).any(axis=1))[0]
    tempM = x[:, k, iM]
    pM = np.where(~np.isnan(tempM).any(axis=1))[0]
    pSLst.append(pS)
    pLLst.append(pL)
    pMLst.append(pM)
    nMat[k, :] = [len(pS), len(pL), len(pM)]

np.where(nMat == 0)
np.sum((np.where(nMat == 0)[1]) == 0)

indKeep = np.where((nMat > 0).all(axis=1))[0]
x = x[:, indKeep, :]
xc = xc[indKeep, :]
yc = yc[indKeep, :]
nMat = nMat[indKeep, :]
pSLst = [pSLst[k] for k in indKeep]
pLLst = [pLLst[k] for k in indKeep]
pMLst = [pMLst[k] for k in indKeep]
jInd = [jInd[k] for k in indKeep]
jInd2 = [jInd2[k] for k in indKeep]
siteIdLst = [siteIdLst[k] for k in jInd]
siteIdLst2 = siteIdLst.copy()
siteIdLst2 = [siteIdLst2[k] for k in jInd2]

# split train and test
jSite, count = np.unique(jInd, return_counts=True)
jSite2, count2 = np.unique(jInd2, return_counts=True)
countAry = np.array([[x, y] for y, x in sorted(zip(count, jSite))])
countAry2 = np.array([[x, y] for y, x in sorted(zip(count2, jSite2))])
nRm = sum(countAry[:, 1] < 5)
nRm2 = sum(countAry2[:, 1] < 5)
indSiteAll = countAry[nRm:, 0].astype(int)
indSiteAll2 = countAry[nRm2:, 0].astype(int)
dictSubset = dict()

k = 0
siteTest2 = indSiteAll2[k::5]
siteTrain2 = np.setdiff1d(indSiteAll2, siteTest2)

indTest2 = np.where(np.isin(jInd2, siteTest2))[0]
indTrain2 = np.where(np.isin(jInd2, siteTrain2))[0]

# correct site index
a1 = [jInd[k] for k in indTest2]
a2 = [jInd2[k] for k in indTest2]
b1 = [jInd[k] for k in indTrain2]
b2 = [jInd2[k] for k in indTrain2]

np.unique(b1).shape
np.unique(b2).shape
np.unique(a1).shape
np.unique(a2).shape

len(set(np.unique(a1)) - set(np.unique(b1)))
len(set(np.unique(b1)) - set(np.unique(a1)))
len(set(np.unique(b1)).intersection(set(np.unique(a1))))

siteInter = list(set(np.unique(b1)).intersection(set(np.unique(a1))))
mat = np.zeros([len(siteInter), 2])
for k, s in enumerate(siteInter):
    c1 = np.sum(b1==s)
    c2 = np.sum(a1==s)
    mat[k, :] = [c1, c2]
    # mat[k, :] = [c1/(c1+c2), c2/(c1+c2)]
mat_sort = mat[np.argsort(mat[:, 0]), :]
# mat_sort=mat
fig, ax = plt.subplots(1, 1)
p1 = ax.bar(np.arange(mat_sort.shape[0]), mat_sort[:, 0], label="train")
p2 = ax.bar(np.arange(mat_sort.shape[0]), mat_sort[:, 1], bottom=mat_sort[:, 0],label="test")

fig.legend()
fig.show()


for k in range(5):
    siteTest = indSiteAll[k::5]
    siteTrain = np.setdiff1d(indSiteAll, siteTest)
    indTest = np.where(np.isin(jInd, siteTest))[0]
    indTrain = np.where(np.isin(jInd, siteTrain))[0]
    dictSubset["testSite_k{}5".format(k)] = siteTest.tolist()
    dictSubset["trainSite_k{}5".format(k)] = siteTrain.tolist()
    dictSubset["testInd_k{}5".format(k)] = indTest.tolist()
    dictSubset["trainInd_k{}5".format(k)] = indTrain.tolist()
