from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt

sn = 1e-5
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


figplot.clickMap(funcM, funcP)

# global feature image
nC = 20
nQ = 20
nD = 20
binC = np.linspace(np.nanmin(DF.c), np.nanmax(DF.c), num=nC + 1)
binQ = np.linspace(np.nanmin(logQ), np.nanmax(logQ), num=nQ + 1)
binD = np.linspace(1, 365, nD + 1)
extent1 = [binD[0], binD[-1], binC[0], binC[-1]]
extent2 = [binQ[0], binQ[-1], binC[0], binC[-1]]
imgLst1 = list()
imgLst2 = list()
for k, siteNo in enumerate(DF.siteNoLst):
    img1 = np.histogram2d(day, DF.c[:, k, 0], bins=[binD, binC], density=True)[0]
    img1 = img1.swapaxes(0, 1)
    img2 = np.histogram2d(logQ[:, k], DF.c[:, k, 0], bins=[binQ, binC], density=True)[0]
    img2 = img2.swapaxes(0, 1)
    imgLst1.append(img1)
    imgLst2.append(img2)
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
    axP2.imshow(imgAry1[:, :, iP], origin='lower', extent=extent1, aspect='auto')
    axP3.scatter(logQ[:, iP], DF.c[:, iP, 0], c=year)
    axP4.imshow(imgAry2[:, :, iP], origin='lower', extent=extent2, aspect='auto')


figplot.clickMap(funcM, funcP)


# PCA of image
from sklearn.decomposition import PCA

m1 = imgAry1.reshape(nC * nD, -1).swapaxes(0, 1)
m2 = imgAry2.reshape(nC * nQ, -1).swapaxes(0, 1)
pca1 = PCA(n_components=150)
pca1.fit(m1)
pca2 = PCA(n_components=150)
pca2.fit(m1)
fig, axes = plt.subplots(2, 1)
axes[0].plot(np.cumsum(pca1.explained_variance_ratio_ * 100))
axes[1].plot(np.cumsum(pca2.explained_variance_ratio_ * 100))
axes[1].set_xlabel('Number of components')
axes[0].set_ylabel('Explained variance')
axes[1].set_ylabel('Explained variance')
fig.show()
