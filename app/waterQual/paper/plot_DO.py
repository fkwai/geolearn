
import matplotlib
# from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

code = '00300'
siteNoLst = dictSite[code]
nSite = len(siteNoLst)
dataName = 'rbWN5'

# load all sequence
dictLSTMLst = list()
# LSTM
label = 'QTFP_C'
dictLSTM = dict()
trainSet = '{}-B10'.format('comb')
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = basins.loadSeq(outName, siteNo)
    dictLSTM[siteNo] = df

# WRTDS
dictWRTDS = dict()
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'Linear-W', 'B10Q', 'output')
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    saveFile = os.path.join(dirWRTDS, siteNo)
    df = pd.read_csv(saveFile, index_col=None).set_index('date')
    # df = utils.time.datePdf(df)
    dictWRTDS[siteNo] = df

# Observation
dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(siteNo, varLst=[code], freq='W')
    dictObs[siteNo] = df


# code = '00010'
# # calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
corrLSTM = np.full([len(siteNoLst), 2], np.nan)
rmseLSTM = np.full([len(siteNoLst),  2], np.nan)
corrWRTDS = np.full([len(siteNoLst), 2], np.nan)
rmseWRTDS = np.full([len(siteNoLst), 2], np.nan)
for k, indT in enumerate([ind1, ind2]):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v0 = dictObs[siteNo][code].iloc[indT].values
        v1 = dictLSTM[siteNo][code].iloc[indT].values
        v2 = dictWRTDS[siteNo][code].iloc[indT].values
        v3 = dictObs[siteNo][code].iloc[indT].values
        [v0, v1, v2], ind = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(v1, v0, rmExt=True)
        rmse2, corr2 = utils.stat.calErr(v2, v0, rmExt=True)
        corrLSTM[indS, k] = corr1
        corrWRTDS[indS, k] = corr2
        rmseLSTM[indS, k] = rmse1
        rmseWRTDS[indS, k] = rmse2

saveFolder = r'C:\Users\geofk\work\paper\waterQuality'
matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 6})
matplotlib.rcParams.update({'legend.fontsize': 12})
# circled map
tempLst = ['11074000', '06902000', '01674500']
siteNoLstCode = dictSite[code]
indS = [siteNoLst.index(siteNo) for siteNo in siteNoLstCode]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']

gs = gridspec.GridSpec(6, 2)
fig = plt.figure(figsize=[12, 6])

axM = fig.add_subplot(gs[:3, 0])
matMap = corrLSTM[indS, 1]
axplot.mapPoint(axM, lat, lon, matMap, s=24)
for siteNo in tempLst:
    xLoc = lon[siteNoLstCode.index(siteNo)]
    yLoc = lat[siteNoLstCode.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='black', fill=False)
    axM.add_patch(circle)
axM.set_title('Testing Correlation of DO')

axM = fig.add_subplot(gs[3:, 0])
matMap = rmseLSTM[indS, 1]
axplot.mapPoint(axM, lat, lon, matMap, s=24)
for siteNo in tempLst:
    xLoc = lon[siteNoLstCode.index(siteNo)]
    yLoc = lat[siteNoLstCode.index(siteNo)]
    circle = plt.Circle([xLoc, yLoc], 1,
                        color='r', fill=False)
    axM.add_patch(circle)
axM.set_title('Testing RMSE of DO')

for k, siteNo in enumerate(tempLst):
    axP = fig.add_subplot(gs[k*2:(k+1)*2, 1])
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfO = waterQuality.readSiteTS(siteNo, [code], freq='W')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM', 'Obs']
    axplot.plotTS(axP, t, [dfL1[code],  dfO[code]],
                  tBar=tBar, sd=sd, styLst='-*', cLst='rk', legLst=legLst)
    iP = siteNoLst.index(siteNo)
    axP.set_title('DO of site {} RMSE={:.2f} corr={:.2f}'.format(
        siteNo, rmseLSTM[iP, 1], corrLSTM[iP, 1]))
    if k == len(tempLst):
        axP.legend()
plt.tight_layout()
fig.show()
fig.savefig(os.path.join(saveFolder, 'plot_{}'.format(code)))
