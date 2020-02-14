from hydroDL.data import gageII
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.post import axplot
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import importlib

# read slope data
dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
doLog = False
if doLog is True:
    mapFolder = os.path.join(dirCQ, 'slopeLogMap')
    boxFolder = os.path.join(dirCQ, 'slopeLogBox')
    dfS = pd.read_csv(os.path.join(dirCQ, 'slopeLog'), dtype={
        'siteNo': str}).set_index('siteNo')
else:
    mapFolder = os.path.join(dirCQ, 'slopeMap')
    boxFolder = os.path.join(dirCQ, 'slopeBox')
    dfS = pd.read_csv(os.path.join(dirCQ, 'slope'), dtype={
        'siteNo': str}).set_index('siteNo')
dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
                  'siteNo': str}).set_index('siteNo')
siteNoLst = dfS.index.tolist()
codeLst = dfS.columns.tolist()

if not os.path.exists(mapFolder):
    os.mkdir(mapFolder)
if not os.path.exists(boxFolder):
    os.mkdir(boxFolder)

# code='00955'
# slopeAry = dfS[code].values
# fig, ax = plt.subplots(1, 1)
# temp = slopeAry[~np.isnan(slopeAry)]
# ax.hist(temp, bins=200, range=[
#         np.percentile(temp, 5), np.percentile(temp, 95)])
# fig.show()

# # plot map
importlib.reload(axplot)
codePdf = waterQuality.codePdf
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)

for code in codeLst:
    slopeAry = dfS[code].values
    nSampleAry = dfN[code].values
    strTitle = 'slope between Q [log ft3/s] and {} [log {}]'.format(
        codePdf['srsName'][code], codePdf['unit'][code])
    strFile = codePdf['shortName'][code]
    ind = np.where((~np.isnan(slopeAry)) & (nSampleAry > 10))[0]
    lat = dfCrd['LAT_GAGE'][ind]
    lon = dfCrd['LNG_GAGE'][ind]
    data = slopeAry[ind]
    vr = np.max([np.abs(np.percentile(data, 5)),
                 np.abs(np.percentile(data, 95))])
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    axplot.mapPoint(ax, lat, lon, data, title=strTitle, vRange=[-vr, vr], s=15)
    fig.show()
    fig.savefig(os.path.join(mapFolder, strFile))

codePdf = waterQuality.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    print(group)
    codeLst = codePdf[codePdf.group == group].index.tolist()
    pos = list(range(0, len(codeLst)))
    for rmExtreme in [True, False]:
        dataLst = list()
        for code in codeLst:
            slopeAry = dfS[code].values
            nSampleAry = dfN[code].values
            ind = np.where((~np.isnan(slopeAry)) & (nSampleAry > 10))[0]
            vr = np.max([np.abs(np.percentile(slopeAry[ind], 10)),
                         np.abs(np.percentile(slopeAry[ind], 90))])
            if rmExtreme is True:
                ind = np.where((~np.isnan(slopeAry)) & (nSampleAry > 10) & (
                    slopeAry <= vr) & (slopeAry >= -vr))[0]
            dataLst.append(slopeAry[ind])
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.violinplot(dataLst, pos, points=500, widths=1,
                      showmeans=True, showextrema=True)
        ax.set_xticks(pos)
        ax.set_xticklabels(codePdf.shortName[codeLst].tolist())
        fig.show()
        if rmExtreme is True:
            fig.savefig(os.path.join(boxFolder, group+'_rmE'))
        else:
            fig.savefig(os.path.join(boxFolder, group))

