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
mapFolder = os.path.join(dirCQ, 'slopeMap')
boxFolder = os.path.join(dirCQ, 'slopeBox')
dfSa = pd.read_csv(os.path.join(dirCQ, 'slope_a'), dtype={
    'siteNo': str}).set_index('siteNo')
dfSb = pd.read_csv(os.path.join(dirCQ, 'slope_b'), dtype={
    'siteNo': str}).set_index('siteNo')
dfCeq = pd.read_csv(os.path.join(dirCQ, 'kate_ceq'), dtype={
    'siteNo': str}).set_index('siteNo')
dfDw = pd.read_csv(os.path.join(dirCQ, 'kate_dw'), dtype={
    'siteNo': str}).set_index('siteNo')
dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
                  'siteNo': str}).set_index('siteNo')
siteNoLst = dfN.index.tolist()
codeLst = dfN.columns.tolist()
dfPLst = [dfSa, dfSb, dfCeq, dfDw]
strPLst = ['slope-a', 'slope-b', 'ceq', 'dw']

if not os.path.exists(mapFolder):
    os.mkdir(mapFolder)
if not os.path.exists(boxFolder):
    os.mkdir(boxFolder)

# code='00955'
# v = dfSb[code].values
# v = dfDw[code].values
# fig, ax = plt.subplots(1, 1)
# temp = v[~np.isnan(v)]
# ax.hist(temp, bins=200, range=[
#         np.percentile(temp, 5), np.percentile(temp, 95)])
# fig.show()

# # plot map
importlib.reload(axplot)
codePdf = waterQuality.codePdf
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)

for code in codeLst:
    for dfP, strP in zip(dfPLst, strPLst):
        pAry = dfP[code].values
        nAry = dfN[code].values
        strTitle = '{} of {} [{}]'.format(
            strP, codePdf['srsName'][code], codePdf['unit'][code])
        strFile = strP+'_'+codePdf['shortName'][code]
        ind = np.where((~np.isnan(pAry)) & (nAry > 10))[0]
        lat = dfCrd['LAT_GAGE'][ind]
        lon = dfCrd['LNG_GAGE'][ind]
        data = pAry[ind]
        vr = np.max([np.abs(np.percentile(data, 5)),
                     np.abs(np.percentile(data, 95))])
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        axplot.mapPoint(ax, lat, lon, data, title=strTitle,
                        vRange=[-vr, vr], s=15)
        # fig.show()
        fig.savefig(os.path.join(mapFolder, strFile))

# plot box
codePdf = waterQuality.codePdf
groupLst = codePdf.group.unique().tolist()
for group in groupLst:
    print(group)
    codeLst = codePdf[codePdf.group == group].index.tolist()
    pos = list(range(0, len(codeLst)))
    for rmExtreme in [True, False]:
        for dfP, strP in zip(dfPLst, strPLst):
            dataLst = list()
            for code in codeLst:
                pAry = dfP[code].values
                nAry = dfN[code].values
                ind = np.where((~np.isnan(pAry)) & (nAry > 10))[0]
                vr = np.max([np.abs(np.percentile(pAry[ind], 10)),
                             np.abs(np.percentile(pAry[ind], 90))])
                if rmExtreme is True:
                    ind = np.where((~np.isnan(pAry)) & (nAry > 10) & (
                        pAry <= vr) & (pAry >= -vr))[0]
                dataLst.append(pAry[ind])
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.violinplot(dataLst, pos, points=500, widths=1,
                          showmeans=True, showextrema=True)
            ax.set_xticks(pos)
            ax.set_xticklabels(codePdf.shortName[codeLst].tolist())
            if rmExtreme is True:
                ax.set_title('{} of {} variables, 10\%-90\%'.format(strP,group))
                fig.savefig(os.path.join(boxFolder, group+'_'+strP+'_rmE'))
            else:
                ax.set_title('{} of {} variables'.format(strP,group))
                fig.savefig(os.path.join(boxFolder, group+'_'+strP))
            # fig.show()

