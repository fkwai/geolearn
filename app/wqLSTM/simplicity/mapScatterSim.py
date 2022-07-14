import matplotlib
import string
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec


# investigate correlation between simlicity and basin attributes.
# remove carbon - less obs, high corr

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)

matRm = count < 200

# load linear/seasonal
dirP = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'
labLst = ['Q', 'S', 'QS']
dictS = dict()
for lab in labLst:
    dirS = dirP.format(lab)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirS, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictS[lab] = matLR

matQ = dictS['Q']
matS = dictS['S']
matQS = dictS['QS']

dirOut = r'C:\Users\geofk\work\waterQuality\paper\G200\simplicity'
outFolder = os.path.join(dirOut, 'mapScatter')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 1.5})
matplotlib.rcParams.update({'lines.markersize': 8})

# map and scatter
for code in codeLst:
    lat, lon = DF.getGeo()
    td = pd.to_datetime(DF.t).dayofyear
    codeStr = usgs.codePdf.loc[code]['shortName']
    indC = codeLst.index(code)
    indS = np.where(~matRm[:, indC])[0]
    td = pd.to_datetime(DF.t).dayofyear

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(2, 5)
    axM1 = mapplot.mapPoint(
        fig, gs[0, :3], lat[indS], lon[indS], matQ[indS, indC])
    axM1.set_title('linearity of {} {}'.format(codeStr, code))
    axM2 = mapplot.mapPoint(
        fig, gs[1, :3], lat[indS], lon[indS], matS[indS, indC])
    axM2.set_title('seasonality of {} {}'.format(codeStr, code))
    axP = fig.add_subplot(gs[:, 3:])
    cs = axP.scatter(matQ[indS, indC], matS[indS, indC], c=matQS[indS, indC])
    cax = plt.colorbar(cs, orientation='vertical')
    cax.set_label('simplicity')
    axP.set_xlabel('linearity of {} {}'.format(codeStr, code))
    axP.set_ylabel('seasonality of {} {}'.format(codeStr, code))
    axP.plot([0, 1], [0, 1], '-k')
    axP.set_aspect(1)
    # plt.tight_layout()
    fig.show()
    fig.savefig(os.path.join(outFolder, 'map_{}'.format(code)))
    fig.savefig(os.path.join(outFolder, 'map_{}.eps'.format(code)))
