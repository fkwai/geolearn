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
c1, c2 = np.nanmin(DF.c[:, :, 0]), np.nanmax(DF.c[:, :, 0])
d1, d2 = 1, 366
q1, q2 = np.nanmin(logQ), np.nanmax(logQ)
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

pcLst = [99, 90, 75]
conG1, conG2, conL1, conL2 = list(), list(), list(), list()
for conLst, img in zip([conG1, conG2, conL1, conL2], [imgG1, imgG2, imgL1, imgL2]):
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
    axT1.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, iP, 1], 'b-')
    # get data
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    xLst, yLst = list(), list()
    for ax, ext, x, y, img, con in zip(
        [axP1, axP2, axP3, axP4],
        [extG1, extG2, extL1[:, iP], extL2[:, iP]],
        [d, q, d, q],
        [c, c, c, c],
        [imgG1[:, :, iP], imgG2[:, :, iP], imgL1[:, :, iP], imgL2[:, :, iP]],
        [conG1[iP], conG2[iP], conL1[iP], conL2[iP]],
    ):
        x1, x2, y1, y2 = ext
        xx = (x - x1) / (x2 - x1) * 100
        yy = (y2 - y) / (y2 - y1) * 100
        ax.imshow(img, extent=[x1, x2, y1, y2], aspect='auto')
        ax.plot(x, y, '*k')
        for k, conK in enumerate(con):
            if conK is None:
                continue
            elif type(conK) is not list:
                conK = [conK]
            for conKI in conK:
                xx = x1 + conKI[:, 1] / 100 * (x2 - x1)
                yy = y2 - conKI[:, 0] / 100 * (y2 - y1)
                ax.plot(xx, yy, '-', color=[1 - k * 0.2, k * 0.2, 0])
        ax.set_xticks([])
        ax.set_yticks([])


figplot.clickMap(funcM, funcP)

# test distance contour
c1 = conG1[89][0]
c2 = conG1[68][0]
fig, ax = plt.subplots(1, 1)
ax.plot(c1[0][:, 1], c1[0][:, 0], 'r-')
ax.plot(c2[0][:, 1], c2[0][:, 0], 'b-')
fig.show()

xx = np.concatenate(c1)
yy = np.concatenate(c2)
dist = np.hypot(*(xx - yy))
import scipy

a = scipy.spatial.distance.cdist(xx, yy)
np.max(np.min(a, axis=1))


# distance contour
ns = len(DF.siteNoLst)
distG1, distG2, distL1, distL2 = [np.zeros([ns, ns, 3]) for x in range(4)]
for iD, (conLst, disMat) in enumerate(
    zip([conG1, conG2, conL1, conL2], [distG1, distG2, distL1, distL2])
):
    for k in range(3):
        print('calculating distance', iD, k)
        for j in range(ns):
            c1 = conLst[j][k]
            for i in range(j):
                c2 = conLst[i][k]
                xx = np.concatenate(c1)
                yy = np.concatenate(c2)
                a = scipy.spatial.distance.cdist(xx, yy)
                dist1 = np.max(np.min(a, axis=1))
                dist2 = np.max(np.min(a, axis=0))
                disMat[j, i, k] = np.max([dist1, dist2])
                disMat[i, j, k] = disMat[j, i, k]
distTup1 = (distG1, distG2, distL1, distL2)

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
distTup2 = (distG1, distG2, distL1, distL2)


# plot distance to an example site

distG1, distG2, distL1, distL2 = distTup1
siteId = '05288705'
indSite = DF.siteNoLst.index(siteId)
indSite = 68
opt = 'G1'

if opt == 'G1':
    distMat = distG1 / 100
elif opt == 'G2':
    distMat = distG2 / 100
elif opt == 'L1':
    distMat = distL1 / 100
elif opt == 'L2':
    distMat = distL2 / 100


def funcM_cont():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(3, 1)
    for k in range(3):
        axM = mapplot.mapPoint(figM, gsM[k, 0], lat, lon, distMat[indSite, :, k])
        axM.plot(lon[indSite], lat[indSite], '*k', ms=20)
    # axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(10, 8))
    gsP = gridspec.GridSpec(2, 3)
    axT1 = figP.add_subplot(gsP[0, :2])
    axT2 = axT1.twinx()
    axT3 = figP.add_subplot(gsP[1, :2])
    axT4 = axT3.twinx()
    axP1 = figP.add_subplot(gsP[0, 2])
    axP2 = figP.add_subplot(gsP[1, 2])
    axPLst = [axT1, axT2, axT3, axT4, axP1, axP2]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon, lat


def funcP_cont(iP, axP):
    print(iP, distMat[indSite, iP, :])
    [axT1, axT2, axT3, axT4, axP1, axP2] = axP
    axT1.plot(DF.t, DF.c[:, indSite, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, indSite, 1], 'b-')
    axT3.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT4.plot(DF.t, DF.q[:, iP, 1], 'b-')
    # get data
    for iS, ax in zip([indSite, iP], [axP1, axP2]):
        d, c, q = utils.rmNan([day, DF.c[:, iS, 0], logQ[:, iS]], returnInd=False)
        y = c
        if opt[-1] == '1':
            x = d
        elif opt[-1] == '2':
            x = q
        if opt == 'L1':
            con = conL1[iS]
            img = imgL1[:, :, iS]
            ext = extL1[:, iS]
        elif opt == 'L2':
            con = conL2[iS]
            img = imgL2[:, :, iS]
            ext = extL2[:, iS]
        elif opt == 'G1':
            con = conG1[iS]
            img = imgG1[:, :, iS]
            ext = extG1
        elif opt == 'G2':
            con = conG2[iS]
            img = imgG2[:, :, iS]
            ext = extG2
        x1, x2, y1, y2 = ext
        xx = (x - x1) / (x2 - x1) * 100
        yy = (y2 - y) / (y2 - y1) * 100
        ax.imshow(img, extent=[x1, x2, y1, y2], aspect='auto')
        ax.plot(x, y, '*k')
        for k, conK in enumerate(con):
            if conK is None:
                continue
            elif type(conK) is not list:
                conK = [conK]
            for conKI in conK:
                xx = x1 + conKI[:, 1] / 100 * (x2 - x1)
                yy = y2 - conKI[:, 0] / 100 * (y2 - y1)
                ax.plot(xx, yy, '-', color=[1 - k * 0.2, k * 0.2, 0])
    d1 = distMat[indSite, iP, 0]
    d2 = distMat[indSite, iP, 1]
    d3 = distMat[indSite, iP, 2]
    ax.set_title('distance = {:.2f} {:.2f} {:.2f}'.format(d1, d2, d3))


figplot.clickMap(funcM_cont, funcP_cont)


from sklearn import manifold
mds_model = manifold.MDS(n_components = 2,dissimilarity = 'precomputed')
mds_fit = mds_model.fit(distMat[:,:,0])
mds_coords = mds_model.fit_transform(distMat[:,:,0]) 

fig,ax=plt.subplots(1,1)
ax.plot(mds_coords[:,0],mds_coords[:,1],'*')
fig.show()

conP = conL1
imgP = imgL1
extP = extL1
i1 = 151
i2 = 60
i3 = 108

distL1[i2, i1]
distL1[i2, i3]
fig, axes = plt.subplots(1, 3)
for iP, ax in zip([i1, i2, i3], axes):
    iP
    ax
    con = conP[iP]
    img = imgP[:, :, iP]
    ext = extP[:, iP]
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    x = d
    y = c
    x1, x2, y1, y2 = ext
    xx = (x - x1) / (x2 - x1) * 100
    yy = (y2 - y) / (y2 - y1) * 100
    ax.imshow(img, extent=[x1, x2, y1, y2], aspect='auto')
    ax.plot(x, y, '*k')
    for k, conK in enumerate(con):
        if conK is None:
            continue
        elif type(conK) is not list:
            conK = [conK]
        for conKI in conK:
            xx = x1 + conKI[:, 1] / 100 * (x2 - x1)
            yy = y2 - conKI[:, 0] / 100 * (y2 - y1)
            ax.plot(xx, yy, '-', color=[1 - k * 0.2, k * 0.2, 0])
fig.show()
