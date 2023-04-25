from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt
from scipy import stats
from hydroDL import utils

sn = 1e-5
code = '00955'
dataName = '{}-B200'.format(code)
DF = dbBasin.DataFrameBasin(dataName)
meanC = np.nanmean(DF.c, axis=0)
lat, lon = DF.getGeo()

tpd = pd.to_datetime(DF.t)
day = tpd.dayofyear
year = tpd.year
logQ = np.log(DF.q[:, :, 1])
logQ[np.isinf(logQ)] = np.nan

# check the distribution of c and q
fig, axes = plt.subplots(1, 2)
axes[0].hist(DF.c[:, :, 0].flatten(), bins=100)
axes[1].hist(logQ.flatten(), bins=100)
fig.show()

n = 100
# global image
c1, c2 = 0, 60
d1, d2 = 1, 365
q1, q2 = -5, 5
dm, cm = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
qm, cm = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
p1 = np.vstack([dm.ravel(), cm.ravel()])
p2 = np.vstack([qm.ravel(), cm.ravel()])
extG1 = [d1, d2, c1, c2]
extG2 = [q1, q2, c1, c2]
imgLst1, imgLst2 = list(), list()
for iP, siteNo in enumerate(DF.siteNoLst):
    print(iP, siteNo)
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    dd = (d - d1) / (d2 - d1)
    cc = (c - c1) / (c2 - c1)
    qq = (q - q1) / (q2 - q1)
    k1 = stats.gaussian_kde([dd, cc])
    k2 = stats.gaussian_kde([qq, cc])
    z1 = np.reshape(k1(p1).T, cm.shape)
    z2 = np.reshape(k2(p2).T, cm.shape)
    imgLst1.append(np.rot90(z1))
    imgLst2.append(np.rot90(z2))
imgG1 = np.stack(imgLst1, axis=-1)
imgG2 = np.stack(imgLst2, axis=-1)

# local image
imgLst1, imgLst2, extLst1, extLst2 = list(), list(), list(), list()
for iP, siteNo in enumerate(DF.siteNoLst):
    print(iP, siteNo)
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    # c1, c2 = np.percentile(c, 5), np.percentile(c, 95)
    c1, c2 = np.min(c), np.max(c)
    d1, d2 = 1, 365
    # q1, q2 = np.percentile(q, 5), np.percentile(q, 95)
    q1, q2 = np.min(q), np.max(q)
    dd = (d - d1) / (d2 - d1)
    cc = (c - c1) / (c2 - c1)
    qq = (q - q1) / (q2 - q1)
    k1 = stats.gaussian_kde([dd, cc])
    k2 = stats.gaussian_kde([qq, cc])
    dm, cm = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
    qm, cm = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
    p1 = np.vstack([dm.ravel(), cm.ravel()])
    p2 = np.vstack([qm.ravel(), cm.ravel()])
    z1 = np.reshape(k1(p1).T, cm.shape)
    z2 = np.reshape(k2(p2).T, cm.shape)
    imgLst1.append(np.rot90(z1))
    imgLst2.append(np.rot90(z2))
    extLst1.append([d1, d2, c1, c2])
    extLst2.append([q1, q2, c1, c2])
imgL1 = np.stack(imgLst1, axis=-1)
imgL2 = np.stack(imgLst2, axis=-1)
extL1 = np.stack(extLst1, axis=-1)
extL2 = np.stack(extLst2, axis=-1)

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
    axP1.plot(day, DF.c[:, iP, 0], 'k*')
    axP1.imshow(imgG1[:, :, iP], extent=extG1, aspect='auto')
    axP2.plot(logQ[:, iP], DF.c[:, iP, 0], 'k*')
    axP2.imshow(imgG2[:, :, iP], extent=extG2, aspect='auto')
    axP3.plot(day, DF.c[:, iP, 0], 'k*')
    axP3.imshow(imgL1[:, :, iP], extent=extL1[:, iP], aspect='auto')
    axP4.plot(logQ[:, iP], DF.c[:, iP, 0], 'k*')
    axP4.imshow(imgL2[:, :, iP], extent=extL2[:, iP], aspect='auto')


figplot.clickMap(funcM, funcP)

# distance
ns = len(DF.siteNoLst)
distG1, distG2, distL1, distL2 = [np.zeros([ns, ns]) for x in range(4)]
for iD, (img, dist) in enumerate(
    zip([imgG1, imgG2, imgL1, imgL2], [distG1, distG2, distL1, distL2])
):
    for j in range(ns):
        for i in range(j):
            # dist[j, i] = np.sqrt(np.mean((img[:, :, j] - img[:, :, i]) ** 2))
            dist[j, i] = np.max(np.abs(img[:, :, j] - img[:, :, i]))
            dist[i, j] = dist[j, i]
fig, ax = plt.subplots(1, 1)
ax.imshow(distG2)
fig.show()

# clustering k-m
from hydroDL.app import cluster
import importlib

importlib.reload(cluster)
outFolder = r'/home/kuai/work/waterQuality/featImage/imageDist'

img = imgL2
matD = distL2
saveStr = 'local-CQ'
saveStrLst = ['global-CT', 'global-CQ', 'local-CT', 'local-CQ']
for img, matD, saveStr in zip(
    [imgG1, imgG2, imgL1, imgL2], [distG1, distG2, distL1, distL2], saveStrLst
):
    saveFolder = os.path.join(outFolder, saveStr)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    kLst, cLst, sLst = list(), list(), list()
    for k in range(2, 10):
        kc, vc = cluster.kmedoid(matD, k)
        figM = plt.figure(figsize=(8, 6))
        gsM = gridspec.GridSpec(1, 1)
        axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, vc, vRange=[0, k - 1])
        axM.set_title('cluster map k={}'.format(k))
        figM.savefig(os.path.join(saveFolder, 'map-k{}'.format(k)))
        kLst.append(kc)
        cLst.append(vc)
        sLst.append(cluster.silhouette(matD, kc, vc))
    fig, ax = plt.subplots(1, 1)
    ax.plot(np.arange(2, 10), sLst)
    ax.set_title('silhouette score')
    ax.set_xlabel('# cluster')
    fig.show()
    fig.savefig(os.path.join(saveFolder, 'silhouette'))

    for nk in range(2, 10):
        kc, vc = cluster.kmedoid(matD, nk)
        imgFolder = os.path.join(saveFolder, 'k{}'.format(nk))
        if not os.path.exists(imgFolder):
            os.mkdir(imgFolder)
        for k in range(nk):
            ind = np.where(vc == k)[0]
            for i in ind:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img[:, :, i])
                figName = 'k{}-i{}'.format(k, i)
                if i in kc:
                    figName = figName + '-center'
                fig.savefig(os.path.join(imgFolder, figName))
                plt.close(fig)

