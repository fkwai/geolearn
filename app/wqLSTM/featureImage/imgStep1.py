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
logQ = np.log(DF.q[:, :, 0])
logQ[np.isinf(logQ)] = np.nan
logC = np.log(DF.c[:, :, 0])
logC[np.isinf(logC)] = np.nan

# check the distribution of c and q
fig, axes = plt.subplots(1, 2)
axes[0].hist(DF.c[:, :, 0].flatten(), bins=100)
axes[1].hist(logQ.flatten(), bins=100)
fig.show()

n = 100
# global image & global min max
c1, c2 = np.nanmin(logC), np.nanmax(logC)
d1, d2 = 1, 365
q1, q2 = np.nanmin(logQ), np.nanmax(logQ)
# xmD, ymD = np.mgrid[0 : 1 : n * 1j, -1 : 2 : 3 * n * 1j]
xmD, ymD = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
xmQ, ymQ = np.mgrid[0 : 1 : n * 1j, 0 : 1 : n * 1j]
pD = np.vstack([xmD.ravel(), ymD.ravel()])
pQ = np.vstack([xmQ.ravel(), ymQ.ravel()])
extGD = [d1, d2, c1, c2]
extGQ = [q1, q2, c1, c2]
imgLstDG, imgLstQG, imgLstDL, imgLstQL = list(), list(), list(), list()
extLst1, extLst2 = list(), list()
for iP, siteNo in enumerate(DF.siteNoLst):
    print(iP, siteNo)
    d, c, q = utils.rmNan([day, logC[:, iP], logQ[:, iP]], returnInd=False)
    dd = (d - d1) / (d2 - d1)
    cG = (c - c1) / (c2 - c1)
    qG = (q - q1) / (q2 - q1)
    # local extent
    cL1, cL2 = np.min(c), np.max(c)
    qL1, qL2 = np.min(q), np.max(q)
    cL = (c - cL1) / (cL2 - cL1)
    qL = (q - qL1) / (qL2 - qL1)
    extLst1.append([d1, d2, cL1, cL2])
    extLst2.append([qL1, qL2, cL1, cL2])
    # C-T
    # xD = np.concatenate([dd - 1, dd, dd + 1])
    # # C-T global
    # yDG = np.concatenate([cG, cG, cG])
    # kDG = stats.gaussian_kde([xD, yDG])
    # zDG = np.rot90(np.reshape(kDG(pD).T, ymD.shape))[n:-n, :]
    # # C-T local
    # yDL = np.concatenate([cL, cL, cL])
    # kDL = stats.gaussian_kde([xD, yDL])
    # zDL = np.rot90(np.reshape(kDL(pD).T, ymD.shape))[n:-n, :]
    
    # C-T global
    kDG = stats.gaussian_kde([dd, cG])
    zDG = np.rot90(np.reshape(kDG(pD).T, ymQ.shape))
    # C-T local
    kDL = stats.gaussian_kde([dd, cL])
    zDL = np.rot90(np.reshape(kDL(pD).T, ymQ.shape))
    # C-Q global
    kQG = stats.gaussian_kde([qG, cG])
    zQG = np.rot90(np.reshape(kQG(pQ).T, ymQ.shape))
    # C-Q local
    kQL = stats.gaussian_kde([qL, cL])
    zQL = np.rot90(np.reshape(kQL(pQ).T, ymQ.shape))
    # save
    imgLstDG.append(zDG)
    imgLstQG.append(zQG)
    imgLstDL.append(zDL)
    imgLstQL.append(zQL)
imgGD = np.stack(imgLstDG, axis=-1)
imgGQ = np.stack(imgLstQG, axis=-1)
imgLD = np.stack(imgLstDL, axis=-1)
imgLQ = np.stack(imgLstQL, axis=-1)
extLD = np.stack(extLst1, axis=-1)
extLQ = np.stack(extLst2, axis=-1)

# test
iP = 95
d, c, q = utils.rmNan([day, logC[:, iP], logQ[:, iP]], returnInd=False)
dd = (d - d1) / (d2 - d1)
cG = (c - c1) / (c2 - c1)
qG = (q - q1) / (q2 - q1)
# local extent
cL1, cL2 = np.min(c), np.max(c)
cL = (c - cL1) / (cL2 - cL1)
# C-T
xD = np.concatenate([dd - 1, dd, dd + 1])
yDL = np.concatenate([cL, cL, cL])
bw=0.25
kDL = stats.gaussian_kde([xD, yDL],bw_method=0.15)
xmD, ymD = np.mgrid[-1 : 2 : 3*n * 1j, 0 : 1 : n * 1j]

pD = np.vstack([xmD.ravel(), ymD.ravel()])

zDL = np.rot90(np.reshape(kDL(pD).T, [n*3,n]))

fig,ax=plt.subplots(1,1)
ax.imshow(zDL,extent=[-1,2,0,1])
ax.plot(xD,yDL,'k*')
fig.show()

fig,ax=plt.subplots(1,1)
ax.plot(xD,yDL,'r*')
fig.show()




# save
outFolder = os.path.join(kPath.dirWQ, 'featImage', 'saveMat')
saveFile = os.path.join(outFolder, 'img_{}.npz'.format(code))
np.savez(
    saveFile,
    imgGD=imgGD,
    imgGQ=imgGQ,
    imgLD=imgLD,
    imgLQ=imgLQ,
    extLD=extLD,
    extLQ=extLQ,
    extGD=extGD,
    extGQ=extGQ,
)

# load
saveFile = os.path.join(outFolder, 'img_{}.npz'.format(code))
npz = np.load(saveFile)
imgGD = npz['imgGD']
imgGQ = npz['imgGQ']
imgLD = npz['imgLD']
imgLQ = npz['imgLQ']
extG1 = npz['extG1']
extG2 = npz['extG2']
extL1 = npz['extL1']
extL2 = npz['extL2']


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
    print(iP, DF.siteNoLst[iP])
    [axT1, axT2, axP1, axP2, axP3, axP4] = axP
    axT1.plot(DF.t, logC[:, iP], 'r*')
    axT2.plot(DF.t, logQ[:, iP], 'b-')
    axP1.plot(day, logC[:, iP], 'k*')
    axP1.imshow(imgGD[:, :, iP], extent=extGD, aspect='auto')
    axP2.plot(logQ[:, iP], logC[:, iP], 'k*')
    axP2.imshow(imgGQ[:, :, iP], extent=extGQ, aspect='auto')
    axP3.plot(day, logC[:, iP], 'k*')
    axP3.imshow(imgLD[:, :, iP], extent=extLD[:, iP], aspect='auto')
    axP4.plot(logQ[:, iP], logC[:, iP], 'k*')
    axP4.imshow(imgLQ[:, :, iP], extent=extLQ[:, iP], aspect='auto')


figplot.clickMap(funcM, funcP)


# contours
from skimage import measure, metrics

pLst = [95, 75]
conGD, conGQ, conLD, conLQ = list(), list(), list(), list()
for conLst, img in zip([conGD, conGQ, conLD, conLQ], [imgGD, imgGQ, imgLD, imgLQ]):
    for k in range(len(DF.siteNoLst)):
        temp = list()
        z = img[:, :, k]
        for pc in pcLst:
            cc = np.percentile(z, pc)
            con = measure.find_contours(z, cc)
            temp.append(con)
        conLst.append(temp)


def funcP(iP, axP):
    print(iP)
    [axT1, axT2, axP1, axP2, axP3, axP4] = axP
    axT1.plot(DF.t, logC[:, iP], 'r*')
    axT2.plot(DF.t, logQ[:, iP], 'b-')
    # get data
    d, c, q = utils.rmNan([day, logC[:, iP], logQ[:, iP]], returnInd=False)
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
    imgLst = 
    imgLst = [imgGD[:, :, iP], imgGQ[:, :, iP], imgLD[:, :, iP], imgLQ[:, :, iP]]
    cirLst = [conGD[iP], conGQ[iP], conLD[iP], conLQ[iP]]
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

# test options for distance


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
                disMat[j, i, k] = metrics.hausdorff_distance(c1, c2, method='modified')
distMat = np.concatenate([disG1, disG2, disL1, disL2], axis=2)
distMat[np.isinf(distMat)] = 0

# clustering k-m

# normalize distMat on each dimension
temp = distMat / distMat.mean(axis=(0, 1))
matD = temp.mean(axis=2)
nM = 3
from hydroDL.app import cluster

center, dist = cluster.kmedoid(distMat[..., -1], 5)
center, dist

# normalize data
for kk, iP in enumerate(center):
    d, c, q = utils.rmNan([day, logC[:, iP], logQ[:, iP]], returnInd=False)
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
