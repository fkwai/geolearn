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


# contours
from skimage import measure, metrics

cG1 = [1, 5, 10]
cG2 = [1, 10, 20]
cL1 = [0.5, 1, 2]
cL2 = [0.5, 1, 2]
conG1, conG2, conL1, conL2 = list(), list(), list(), list()
for conLst, img, cLst in zip(
    [conG1, conG2, conL1, conL2], [imgG1, imgG2, imgL1, imgL2], [cG1, cG2, cL1, cL2]
):
    for k in range(len(DF.siteNoLst)):
        temp = list()
        z = img[:, :, k]
        for cc in cLst:
            con = measure.find_contours(z, cc)
            temp.append(con)
        conLst.append(temp)


def funcP(iP, axP):
    print(iP)
    [axT1, axT2, axP1, axP2, axP3, axP4] = axP
    axT1.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, iP, 1], 'b-')
    # get data
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    xLst, yLst = list(), list()
    for ext, x, y in zip([extG1, extG2, extL1, extL2], [d, q, d, q], [c, c, c, c]):
        if type(ext) is list:
            xx = (x - ext[0]) / (ext[1] - ext[0]) * 100
            yy = (1 - (y - ext[2]) / (ext[3] - ext[2])) * 100
        else:
            xx = (x - ext[0, iP]) / (ext[1, iP] - ext[0, iP]) * 100
            yy = (1 - (y - ext[2, iP]) / (ext[3, iP] - ext[2, iP])) * 100
        xLst.append(xx)
        yLst.append(yy)
    imgLst = [imgG1[:, :, iP], imgG2[:, :, iP], imgL1[:, :, iP], imgL2[:, :, iP]]
    cirLst = [conG1[iP], conG2[iP], conL1[iP], conL2[iP]]
    for k, ax in enumerate([axP1, axP2, axP3, axP4]):
        ax.imshow(imgLst[k])
        ax.plot(xLst[k], yLst[k], '*k')
        for cir in cirLst[k]:
            if cir is not None:
                if type(cir) is list:
                    for con in cir:
                        ax.plot(con[:, 1], con[:, 0], '-r')
                else:
                    ax.plot(cir[:, 1], cir[:, 0], '-r')


figplot.clickMap(funcM, funcP)

# distance
ns = len(DF.siteNoLst)
disG1, disG2, disL1, disL2 = [np.zeros([ns, ns, 3]) for x in range(4)]
for iD, (conLst, disMat) in enumerate(
    zip([conG1, conG2, conL1, conL2], [disG1, disG2, disL1, disL2])
):
    for k in range(3):
        print(iD, k)
        for j in range(ns):
            c1 = conLst[j][k]
            if type(c1) is list and len(c1) > 0:
                c1 = c1[np.argmax([len(x) for x in c1])]
            for i in range(ns):
                c2 = conLst[i][k]
                if type(c2) is list and len(c2) > 0:
                    c2 = c2[np.argmax([len(x) for x in c2])]
                disMat[j, i, k] = metrics.hausdorff_distance(c1, c2)
distMat = np.concatenate([disG1, disG2, disL1, disL2], axis=2)
distMat[np.isinf(distMat)] = 0

# clustering k-m

# normalize distMat on each dimension
temp = distMat / distMat.mean(axis=(0, 1))
matD = temp.mean(axis=2)
nM = 3
from hydroDL.app import cluster
center,dist=cluster.kmedoid(distMat[...,-1],5)
center, dist

# normalize data
for kk, iP in enumerate(center):
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    xLst, yLst = list(), list()
    for ext, x, y in zip([extG1, extG2, extL1, extL2], [d, q, d, q], [c, c, c, c]):
        if type(ext) is list:
            xx = (x - ext[0]) / (ext[1] - ext[0]) * 100
            yy = (1 - (y - ext[2]) / (ext[3] - ext[2])) * 100
        else:
            xx = (x - ext[0, iP]) / (ext[1, iP] - ext[0, iP]) * 100
            yy = (1 - (y - ext[2, iP]) / (ext[3, iP] - ext[2, iP])) * 100
        xLst.append(xx)
        yLst.append(yy)
    imgLst = [imgG1[:, :, iP], imgG2[:, :, iP], imgL1[:, :, iP], imgL2[:, :, iP]]
    cirLst = [conG1[iP], conG2[iP], conL1[iP], conL2[iP]]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k, ax in enumerate(axes):
        ax.imshow(imgLst[k])
        ax.plot(xLst[k], yLst[k], '*k')
        for cir in cirLst[k]:
            if cir is not None:
                if type(cir) is list:
                    for con in cir:
                        ax.plot(con[:, 1], con[:, 0], '-r')
                else:
                    ax.plot(cir[:, 1], cir[:, 0], '-r')
    fig.show()
    fig.suptitle('cluster {}'.format(kk))

figM = plt.figure(figsize=(8, 6))
gsM = gridspec.GridSpec(1, 1)
axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, vCluster)
figM.show()

matD[center, 144]
