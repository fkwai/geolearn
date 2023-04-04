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
logQ = np.log(DF.q[:, :, 1])
logQ[np.isinf(logQ)] = np.nan

# global feature image
from scipy import stats
from hydroDL import utils

# check the distribution of c and q
fig, axes = plt.subplots(1, 2)
axes[0].hist(DF.c[:, :, 0].flatten(), bins=100)
axes[1].hist(logQ.flatten(), bins=100)
fig.show()

n = 100
# global image
c1, c2 = 0, 60
d1, d2 = 1, 365
q1, q2 = -10, 5
dm, cm = np.mgrid[d1 : d2 : n * 1j, c1 : c2 : n * 1j]
qm, cm = np.mgrid[q1 : q2 : n * 1j, c1 : c2 : n * 1j]
p1 = np.vstack([dm.ravel(), cm.ravel()])
p2 = np.vstack([qm.ravel(), cm.ravel()])
extG1 = [d1, d2, c1, c2]
extG2 = [q1, q2, c1, c2]
imgLst1, imgLst2 = list(), list()
for iP, siteNo in enumerate(DF.siteNoLst):
    print(iP, siteNo)
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    k1 = stats.gaussian_kde([d, c])
    k2 = stats.gaussian_kde([q, c])
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
    k1 = stats.gaussian_kde([d, c])
    k2 = stats.gaussian_kde([q, c])
    c1, c2 = np.percentile(c, 5), np.percentile(c, 95)
    d1, d2 = 1, 365
    q1, q2 = np.percentile(q, 5), np.percentile(q, 95)
    dm, cm = np.mgrid[d1 : d2 : n * 1j, c1 : c2 : n * 1j]
    qm, cm = np.mgrid[q1 : q2 : n * 1j, c1 : c2 : n * 1j]
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


# fig, axes = plt.subplots(1, 2)
# axes[0].hist(imgG1.flatten(), bins=100)
# axes[1].hist(imgG2.flatten(), bins=100)
# fig.show()

# from skimage import measure, metrics

import random

img = imgG1[:, :, random.randint(0, 155)]
cLst = [0.05]
fig, ax = plt.subplots()
contours = measure.find_contours(img, cLst[0])
ax.imshow(img)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], '-k')
fig.show()

# contours
from skimage import measure, metrics

# C-T
cLst1 = [0.05, 0.01]
conLst1 = list()
for k in range(len(DF.siteNoLst)):
    temp = list()
    imgLst = [imgG1[:, :, k], imgL1[:, :, k]]
    for img in imgLst:
        tempC = list()
        for cc in cLst:
            con = measure.find_contours(img, cc/1e2)
            tempC.append(con)
        temp.append(tempC)
    conLst1.append(temp)

# C-Q
cLst2 = [0.05, 0.01]
conLst2 = list()
for k in range(len(DF.siteNoLst)):
    temp = list()
    imgLst = [imgG2[:, :, k], imgL2[:, :, k]]
    for img in imgLst:
        tempC = list()
        for cc in cLst:
            con = measure.find_contours(img, cc)
            tempC.append(con)
        temp.append(tempC)
    conLst2.append(temp)


def funcP(iP, axP):
    print(iP)
    [axT1, axT2, axP1, axP2, axP3, axP4] = axP
    axT1.plot(DF.t, DF.c[:, iP, 0], 'r*')
    axT2.plot(DF.t, DF.q[:, iP, 1], 'b-')
    # CT Global    
    axP1.imshow(imgG1[:, :, iP])
    for cons in conLst1[iP][0]:
        for contour in cons:
            if len(contour)>0:
                axP1.plot(contour[:, 1], contour[:, 0], '-k')
    # CQ Global    
    axP2.imshow(imgG2[:, :, iP])
    for cons in conLst2[iP][0]:
        for contour in cons:
            if len(contour)>0:
                axP2.plot(contour[:, 1], contour[:, 0], '-k')
    # CT local
    axP3.imshow(imgL1[:, :, iP])
    for cons in conLst1[iP][1]:
        for contour in cons:
            if len(contour)>0:
                axP3.plot(contour[:, 1], contour[:, 0], '-k')
    # CQ local
    axP4.imshow(imgL2[:, :, iP])
    for cons in conLst2[iP][1]:
        for contour in cons:
            if len(contour)>0:
                axP4.plot(contour[:, 1], contour[:, 0], '-k')



figplot.clickMap(funcM, funcP)


fig, ax = plt.subplots()
contours = measure.find_contours(img, 0.002)
ax.imshow(img)
for contour in contours:
    ax.plot(contour[:, 1], contour[:, 0], '-k')
fig.show()

c1 = measure.find_contours(img1, 0.05)
img2 = imgG2[:, :, 1]
c2 = measure.find_contours(img2, 0.05)
metrics.hausdorff_distance(c1, c2)
