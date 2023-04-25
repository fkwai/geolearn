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

# contours
from skimage import measure, metrics

pcLst = [75, 90, 99]
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

# distance
# distance contour
import scipy
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
distMat = np.concatenate([distG1, distG2, distL1, distL2], axis=2)
distMat[np.isinf(distMat)] = 0


# clustering k-m
from hydroDL.app import cluster
import importlib

importlib.reload(cluster)
outFolder = r'/home/kuai/work/waterQuality/featImage/contourDist'

img = imgL2
matD = distL2
saveStr = 'local-CQ'
saveStrLst = ['global-CT', 'global-CQ', 'local-CT', 'local-CQ']
for img, matDR, conLst, saveStr in zip(
    [imgG1, imgG2, imgL1, imgL2],
    [distG1, distG2, distL1, distL2],
    [conG1, conG2, conL1, conL2],
    saveStrLst,
):
    saveFolder = os.path.join(outFolder, saveStr)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    kLst, cLst, sLst = list(), list(), list()
    matD = (matDR / matDR.mean(axis=(0, 1))).mean(axis=2)
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

    for nk in range(2, 5):
        kc, vc = cluster.kmedoid(matD, nk)
        imgFolder = os.path.join(saveFolder, 'k{}'.format(nk))
        if not os.path.exists(imgFolder):
            os.mkdir(imgFolder)
        for k in range(nk):
            ind = np.where(vc == k)[0]
            for i in ind:
                fig, ax = plt.subplots(1, 1)
                ax.imshow(img[:, :, i])
                for icon, con in enumerate(conLst[i]):
                    for cc in con:
                        if type(cc) is not list:
                            cc = [cc]
                        for ccc in cc:
                            ax.plot(
                                ccc[:, 1],
                                ccc[:, 0],
                                '-',
                                color=[1 - 0.2 * icon, 0.2 * icon, 0],
                            )
                figName = 'k{}-i{}'.format(k, i)
                if i in kc:
                    figName = figName + '-center'
                fig.savefig(os.path.join(imgFolder, figName))
                plt.close(fig)
