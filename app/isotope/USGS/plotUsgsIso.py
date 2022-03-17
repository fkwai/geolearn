import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
from hydroDL.data import usgs, gageII
from hydroDL import kPath
import pandas as pd
import numpy as np
import time
import os

# site inventory
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLstIso = ['82085', '82745', '82082']
t0 = time.time()
sd = pd.datetime(1982, 1, 1)
siteNoLst = list()
countLst = list()
dfLst = list()
for i, siteNo in enumerate(siteNoLstAll):
    print(i)
    dfC, dfCF = usgs.readSample(
        siteNo, codeLst=codeLstIso, flag=2, startDate=sd, csv=True)
    dfC = dfC.dropna(how='all')
    if len(dfC) > 0:
        dfLst.append(dfC)
        siteNoLst.append(siteNo)
        countLst.append(len(dfC))
values, counts = np.unique(countLst, return_counts=True)
pd.DataFrame(index=values, columns=['nSite'], data=counts)

# plot selected sites
countThe = 100
siteNoSel = list()
countSel = list()
dfSel = list()
for i, siteNo in enumerate(siteNoLst):
    if countLst[i] > countThe:
        dfSel.append(dfLst[i])
        siteNoSel.append(siteNoLst[i])
        countSel.append(countLst[i])
len(siteNoSel)

# tsmap
varLst = ['LAT_GAGE', 'LNG_GAGE']
dfGeo = gageII.readData(varLst=varLst, siteNoLst=siteNoSel)
lat = dfGeo['LAT_GAGE'].values
lon = dfGeo['LNG_GAGE'].values


def funcM():
    figM = plt.figure(figsize=(12, 5))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, np.array(countSel))
    axM.set_title('Count of Isotope obs, site with >{} obs'.format(countThe))
    figP, axP = plt.subplots(3, 1, figsize=(12, 4))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    dfC = dfSel[iP]
    axplot.plotTS(axP[0], dfC.index, dfC['82085'])
    axplot.plotTS(axP[1], dfC.index, dfC['82745'])
    axplot.plotTS(axP[2], dfC.index, dfC['82082'])
    strTitle = ('{} {}'.format(siteNoSel[iP], countSel[iP]))
    axP[0].set_title(strTitle)


figM, figP = figplot.clickMap(funcM, funcP)
