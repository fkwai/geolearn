from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json

# ts map of single dataset, label and code
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')


# select number of sites
countS = np.sort(dfRes1['count'].values)[::-1]
fig, ax = plt.subplots(1, 1)
ax.plot(np.arange(len(countS)), countS, '-*')
# ax.set_yscale('log')
# ax.set_xscale('log')
fig.show()

# plot map
nS = 200
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]

# crd
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values


def funcMap():
    figM, axM = plt.subplots(1, 3, figsize=(6, 12))
    mat1 = dfR1['corr'].values
    mat2 = dfR2['corr'].values
    axplot.mapPoint(axM[0], lat, lon, mat1, vRange=[0, 1], s=16)
    axplot.mapPoint(axM[1], lat, lon, mat2, vRange=[0, 1], s=16)
    axplot.mapPoint(axM[2], lat, lon, mat1**2/mat2**2, vRange=[0.5, 1.5], s=16)
    shortName = usgs.codePdf.loc[code]['shortName']
    axM[0].set_title('WRTDS corr, {}'.format(shortName))
    axM[1].set_title('T only corr, {}'.format(shortName))
    axM[2].set_title('R2 ratio, {}'.format(shortName))
    figP, axP = plt.subplots(1, 1, figsize=(16, 4))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')[code]
    t = dfO.index
    file1 = os.path.join(dirRoot1, 'output', siteNo)
    file2 = os.path.join(dirRoot2, 'output', siteNo)
    dfP1 = pd.read_csv(file1, index_col='date')[code]
    dfP2 = pd.read_csv(file2, index_col='date')[code]
    v = [dfP1.values, dfP2.values, dfO.values]
    [v1, v2, o], iv = utils.rmNan([dfP1.values, dfP2.values, dfO.values])
    tt = t[iv]
    styLst = [['-*'] for x in range(3)]
    axplot.plotTS(axP, tt.values, [v1, v2, o], cLst='rbk')
    # print corr
    rmse1, corr1 = utils.stat.calErr(v[0], v[-1])
    rmse2, corr2 = utils.stat.calErr(v[1], v[-1])
    axP.set_title('site {} WRTDS {:.2f} only T {:.2f}'.format(
        siteNo, corr1, corr2))


figM, figP = figplot.clickMap(funcMap, funcPoint)
