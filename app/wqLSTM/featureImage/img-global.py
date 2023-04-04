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
from scipy import stats
from hydroDL import utils

# check the distribution of c and q
fig, axes = plt.subplots(1, 2)
axes[0].hist(DF.c[:, :, 0].flatten(), bins=100)
axes[1].hist(logQ.flatten(), bins=100)
fig.show()

n = 100
c1, c2 = 0, 60
d1, d2 = 1, 365
q1, q2 = -10, 5
dm, cm = np.mgrid[d1 : d2 : n * 1j, c1 : c2 : n * 1j]
qm, cm = np.mgrid[q1 : q2 : n * 1j, c1 : c2 : n * 1j]
p1 = np.vstack([dm.ravel(), cm.ravel()])
p2 = np.vstack([qm.ravel(), cm.ravel()])
extent1 = [d1, d2, c1, c2]
extent2 = [q1, q2, c1, c2]
imgLst1 = list()
imgLst2 = list()
for iP, siteNo in enumerate(DF.siteNoLst):
    print(iP, siteNo)
    d, c, q = utils.rmNan([day, DF.c[:, iP, 0], logQ[:, iP]], returnInd=False)
    k1 = stats.gaussian_kde([d, c])
    k2 = stats.gaussian_kde([q, c])
    z1 = np.reshape(k1(p1).T, cm.shape)
    z2 = np.reshape(k2(p2).T, cm.shape)
    imgLst1.append(np.rot90(z1))
    imgLst2.append(np.rot90(z2))
imgAry1 = np.stack(imgLst1, axis=-1)
imgAry2 = np.stack(imgLst2, axis=-1)


# CQ CT plot to image
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
    axP1.plot(day, DF.c[:, iP, 0], 'k*')
    axP1.imshow(imgAry1[:, :, iP], extent=extent1, aspect='auto')
    axP2.plot(logQ[:, iP], DF.c[:, iP, 0], 'k*')
    axP2.imshow(imgAry2[:, :, iP], extent=extent2, aspect='auto')


figplot.clickMap(funcM, funcP)
