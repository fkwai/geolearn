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

# load
outFolder = os.path.join(kPath.dirWQ, 'featImage', 'saveMat')
saveFile = os.path.join(outFolder, 'img_{}.npz'.format(code))
npz = np.load(saveFile)
imgGD = npz['imgGD']
imgGQ = npz['imgGQ']
imgLD = npz['imgLD']
imgLQ = npz['imgLQ']
extGD = npz['extGD']
extGQ = npz['extGQ']
extLD = npz['extLD']
extLQ = npz['extLQ']


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

pcLst = [99, 90, 75]
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
    for ext, x, y in zip([extGD, extGQ, extLD, extLQ], [d, q, d, q], [c, c, c, c]):
        if len(ext.shape) == 1:
            xx = (x - ext[0]) / (ext[1] - ext[0]) * 100
            yy = (1 - (y - ext[2]) / (ext[3] - ext[2])) * 100
        else:
            xx = (x - ext[0, iP]) / (ext[1, iP] - ext[0, iP]) * 100
            yy = (1 - (y - ext[2, iP]) / (ext[3, iP] - ext[2, iP])) * 100
        xLst.append(xx)
        yLst.append(yy)
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

# load LSTM performance
from hydroDL.master import basinFull

label = 'QFT2C'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'
ep = 500
outName = '{}-{}-{}'.format(dataName, label, trainSet)
yP2, ycP2 = basinFull.testModel(outName, testSet=testSet, ep=ep)
matObs = DF.extractT([code])
obs2 = DF.extractSubset(matObs, testSet)
corr = utils.stat.calCorr(yP2[:-1, :, :], obs2)
from hydroDL.app.waterQuality import WRTDS
yW2 = WRTDS.testWRTDS(dataName, trainSet, testSet, [code])
corrW = utils.stat.calCorr(yW2, obs2)

# CQ CT plot to image
def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, corr)
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


figplot.clickMap(funcM, funcP)

# save top 25 and bottom 25
# indCorr = np.argsort(corr.flatten())
# for iP in indCorr[:25]:
#     folder = r'/home/kuai/work/waterQuality/featImage/vsLSTM/{}/bad'.format(code)

# for iP in indCorr[-25:]:
#     folder = r'/home/kuai/work/waterQuality/featImage/vsLSTM/{}/good'.format(code)
#     siteNo = DF.siteNoLst[iP]
#     d, c, q = utils.rmNan([day, logC[:, iP], logQ[:, iP]], returnInd=False)
#     xLst, yLst = list(), list()
#     for ext, x, y in zip([extGD, extGQ, extLD, extLQ], [d, q, d, q], [c, c, c, c]):
#         if len(ext.shape) == 1:
#             xx = (x - ext[0]) / (ext[1] - ext[0]) * 100
#             yy = (1 - (y - ext[2]) / (ext[3] - ext[2])) * 100
#         else:
#             xx = (x - ext[0, iP]) / (ext[1, iP] - ext[0, iP]) * 100
#             yy = (1 - (y - ext[2, iP]) / (ext[3, iP] - ext[2, iP])) * 100
#         xLst.append(xx)
#         yLst.append(yy)
#     for figLabel, img, con, x, y in zip(
#         ['GD', 'GQ', 'LD', 'LQ'],
#         [imgGD, imgGQ, imgLD, imgLQ],
#         [conGD, conGQ, conLD, conLQ],
#         xLst,
#         yLst,
#     ):
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(x, y, 'k*')
#         ax.imshow(img[:, :, iP])
#         for cir in con[iP]:
#             if type(cir) is list:
#                 for con in cir:
#                     ax.plot(con[:, 1], con[:, 0], '-r')
#             else:
#                 ax.plot(cir[:, 1], cir[:, 0], '-r')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         # fig.show()
#         saveFolder = os.path.join(folder, figLabel)
#         if not os.path.exists(saveFolder):
#             os.makedirs(saveFolder)
#         fig.savefig(
#             os.path.join(saveFolder, '{:.0f}_{}'.format(corr[iP, 0] * 100, siteNo))
#         )
#         fig.clf()


# distance contour
import scipy

ns = len(DF.siteNoLst)
distG1, distG2, distL1, distL2 = [np.zeros([ns, ns, 3]) for x in range(4)]
for iD, (conLst, disMat) in enumerate(
    zip([conGD, conGQ, conLD, conLQ], [distG1, distG2, distL1, distL2])
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
    zip([imgGD, imgGQ, imgLD, imgLQ], [distG1, distG2, distL1, distL2])
):
    for j in range(ns):
        for i in range(j):
            # dist[j, i] = np.sqrt(np.mean((img[:, :, j] - img[:, :, i]) ** 2))
            dist[j, i] = np.max(np.abs(img[:, :, j] - img[:, :, i]))
            dist[i, j] = dist[j, i]
distTup2 = (distG1, distG2, distL1, distL2)


# clustering k-m
from hydroDL.app import cluster
from sklearn import manifold

opt = 'LD'
optDist = 0
if optDist == 0:
    distG1, distG2, distL1, distL2 = distTup1
elif optDist == 1:
    distG1, distG2, distL1, distL2 = distTup2

if opt == 'GD':
    distMat = distG1 / 100
elif opt == 'GQ':
    distMat = distG2 / 100
elif opt == 'LD':
    distMat = distL1 / 100
elif opt == 'LQ':
    distMat = distL2 / 100

distMat = distMat.mean(axis=2)
nk = 3
kc, vc = cluster.kmedoid(distMat, nk)

mds_model = manifold.MDS(n_components=2, dissimilarity='precomputed')
mds_fit = mds_model.fit(distMat)
mds_coords = mds_model.fit_transform(distMat)
mds_model.stress_

fig, ax = plt.subplots(1, 1)
for k in range(nk):
    ax.plot(mds_coords[vc == k, 0], mds_coords[vc == k, 1], '*')
fig.show()

# corr plot

fig, ax = plt.subplots(1, 1)
for k in range(nk):
    ax.plot(corr[vc == k, 0], corrW[vc == k, 0], '*')
fig.show()
