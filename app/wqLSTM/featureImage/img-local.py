from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt

sn = np.exp(-5)
code = '00955'
dataName = '{}-B200'.format(code)
DF = dbBasin.DataFrameBasin(dataName)


meanC = np.nanmean(DF.c, axis=0)
lat, lon = DF.getGeo()

tpd = pd.to_datetime(DF.t)
day = tpd.dayofyear
year = tpd.year
logQ = np.log(DF.q[:, :, 1] + sn)

# CQ CT plot
def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, meanC)
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(10, 8))
    gsP = gridspec.GridSpec(2, 2)
    axT1 = figP.add_subplot(gsP[0, :])
    axT2 = axT1.twinx()
    axP1 = figP.add_subplot(gsP[1, 0])
    axP2 = figP.add_subplot(gsP[1, 1])
    axPLst = [axT1, axT2, axP1, axP2]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    [axT1, axT2, axP1, axP2] = axP
    axT1.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, iP, 1], 'b-')
    sc1 = axP1.scatter(day, DF.c[:, iP, 0], c=year)
    sc2 = axP2.scatter(logQ[:, iP], DF.c[:, iP, 0], c=year)
    # plt.colorbar(sc1)


figplot.clickMap(funcM, funcP)

# local feature image range
nC = 50
nQ = 50
nD = 50
binD = np.linspace(1, 365, nD + 1)
imgLst1 = list()
imgLst2 = list()
extLst1 = list()
extLst2 = list()
for k, siteNo in enumerate(DF.siteNoLst):
    c = DF.c[:, k, 0]
    q = logQ[:, k]
    ind = ~np.isnan(c) & ~np.isnan(q)
    c = c[ind]
    q = q[ind]
    d = day[ind]
    binC = np.linspace(np.nanmin(c), np.nanmax(c), num=nC + 1)
    binQ = np.linspace(np.nanmin(q), np.nanmax(q), num=nQ + 1)
    extent1 = [binD[0], binD[-1], binC[0], binC[-1]]
    extent2 = [binQ[0], binQ[-1], binC[0], binC[-1]]
    img1 = np.histogram2d(day, DF.c[:, k, 0], bins=[binD, binC])[0]
    img1 = img1.swapaxes(0, 1)
    img2 = np.histogram2d(logQ[:, k], DF.c[:, k, 0], bins=[binQ, binC])[0]
    img2 = img2.swapaxes(0, 1)
    imgLst1.append(img1)
    imgLst2.append(img2)
    extLst1.append(extent1)
    extLst2.append(extent2)
imgAry1 = np.stack(imgLst1, axis=-1)
imgAry2 = np.stack(imgLst2, axis=-1)


# CQ CT plot to image
def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, meanC)
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(10, 8))
    gsP = gridspec.GridSpec(2, 4)
    axT1 = figP.add_subplot(gsP[0, :])
    axT2 = axT1.twinx()
    axP1 = figP.add_subplot(gsP[1, 0])
    axP2 = figP.add_subplot(gsP[1, 1])
    axP3 = figP.add_subplot(gsP[1, 2])
    axP4 = figP.add_subplot(gsP[1, 3])
    axPLst = [axT1, axT2, axP1, axP2, axP3, axP4]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    [axT1, axT2, axP1, axP2, axP3, axP4] = axP
    axT1.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, iP, 1], 'b-')
    axP1.scatter(day, DF.c[:, iP, 0], c=year)
    im1=axP2.imshow(imgAry1[:, :, iP], origin='lower', extent=extLst1[iP], aspect='auto')
    axP3.scatter(logQ[:, iP], DF.c[:, iP, 0], c=year)
    im2 = axP4.imshow(
        imgAry2[:, :, iP], origin='lower', extent=extLst2[iP], aspect='auto'
    )
    # plt.colorbar(im1)
    # plt.colorbar(im2)


figplot.clickMap(funcM, funcP)


# PCA of image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

mA1 = imgAry1.reshape(nC * nD, -1).swapaxes(0, 1)
mA2 = imgAry2.reshape(nC * nQ, -1).swapaxes(0, 1)
scaler = StandardScaler()
m1 = scaler.fit_transform(mA1)
m2 = scaler.fit_transform(mA2)


pca1 = PCA(n_components=150)
pc1 = pca1.fit(m1)
p1 = pca1.transform(m1)
pca2 = PCA(n_components=150)
pca2.fit(m2)
p2 = pca1.transform(m2)

fig, axes = plt.subplots(2, 1)
axes[0].plot(np.cumsum(pca1.explained_variance_ratio_ * 100))
axes[1].plot(np.cumsum(pca2.explained_variance_ratio_ * 100))
axes[1].set_xlabel('Number of components')
axes[0].set_ylabel('Explained variance')
axes[1].set_ylabel('Explained variance')
fig.show()

# eigen image
nE = 5
e1 = pca1.components_[:nE].reshape(nE,nD, nC)
e2 = pca2.components_[:nE].reshape(nE,nQ, nC)
fig, axes = plt.subplots(2, nE)
for k in range(nE):
    axes[0, k].imshow(e1[k, ::-1, :])
    axes[1, k].imshow(e2[k, ::-1, :])
fig.show()

# load perfomance of LSTM
from hydroDL.master import basinFull
from hydroDL import kPath, utils

label = 'QT2C'
dataName = '{}-{}'.format(code, 'B200')
ep = 500
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
matObs = DF.extractT([code])
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
outName = '{}-{}-{}'.format(dataName, label, trainSet)
dictMaster = basinFull.loadMaster(outName)
yP1, ycP1 = basinFull.testModel(outName, DF=DF, testSet=trainSet, ep=ep)
yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=ep)
if len(dictMaster['varY']) > 1:
    yP1 = yP1[:, :, 1:]
    yP2 = yP2[:, :, 1:]
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)

i = 0
j = 0
fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes[0].scatter(p1[:, i], p1[:, j], c=corrL2)
axes[1].scatter(p2[:, i], p2[:, j], c=corrL2)
fig.suptitle('pc{} vs pc{}'.format(i, j))
fig.show()

r1 = np.ndarray(50)
r2 = np.ndarray(50)
for k in range(50):
    r1[k] = np.corrcoef(p1[:, k], corrL2[:, 0])[0, 1]
    r2[k] = np.corrcoef(p2[:, k], corrL2[:, 0])[0, 1]


i1, j1 = np.argsort(r1**2)[-2:]
i2, j2 = np.argsort(r2**2)[-2:]
fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes[0].scatter(p1[:, i1], p1[:, j1], c=corrL2)
axes[1].scatter(p2[:, i2], p2[:, j2], c=corrL2)
fig.suptitle('pc{} vs pc{}; pc{} vs pc{}'.format(i1, j1, i2, j2))
fig.show()

i1=1
j1=1
fig, axes = plt.subplots(2, 1, figsize=(4, 8))
axes[0].scatter(p1[:, i1], corrL2, c=corrL2)
axes[1].scatter(p2[:, j1], corrL2, c=corrL2)
fig.suptitle('pc{} vs LSTM R; pc{} vs LSTM R'.format(i1, j1))
fig.show()