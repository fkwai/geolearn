
import matplotlib
# from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality as wq
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
from astropy.timeseries import LombScargle


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
if False:
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, codeLst)
    corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)
    # load basin attributes
    dfG = gageII.readData(siteNoLst=siteNoLst)
    dfG = gageII.updateRegion(dfG)
    dfG = gageII.updateCode(dfG)

t = dictObs[siteNoLst[0]].index.values
tt = np.datetime64('2010-01-01')
t0 = np.datetime64('1980-01-01')
indT1 = np.where((t < tt) & (t >= t0))[0]
indT2 = np.where(t >= tt)[0]

# matplotlib.rcParams.update({'font.size': 18})
# matplotlib.rcParams.update({'lines.linewidth': 2})
# matplotlib.rcParams.update({'lines.markersize': 12})

# 121 scatter

# calculate LombScargle
pMat = np.full([len(siteNoLst), len(codeLst)], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        df = dictObs[siteNo]
        t = np.arange(len(df))*7
        y = df[code]
        tt, yy = utils.rmNan([t, y], returnInd=False)
        p = LombScargle(tt, yy).power(1/365)
        pMat[indS, ic] = p

# ts map
dictPlot = dict()
dictPlot['00010'] = ['02338500', '12323770', '11141050']
dictPlot['00300'] = ['01668000', '05288705', '06426500']
dictPlot['00915'] = ['09371492', '05465500', '02175000']
dictPlot['00955'] = ['10343500', '05465500', '10172200']
dictPlot['00600'] = ['01118500', '10336698', '02215500']
dictPlot['00600'] = ['01118500', '10336698', '02215500']
dictPlot['00405'] = ['06324500', '06313500', '01196500']

for code in dictPlot.keys():
    siteNoCode = dictSite[code]
    siteNoP = dictPlot[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    ic = codeLst.index(code)
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoCode)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    shortName = usgs.codePdf.loc[code]['shortName']

    fig = plt.figure(figsize=[12, 9])
    gsM = gridspec.GridSpec(4, 5)

    ax = fig.add_subplot(gsM[0, 0])
    sc = axplot.scatter121(ax, corrMat[indS, ic, 1],
                           corrMat[indS, ic, 2], pMat[indS, ic])
    fig.colorbar(sc, ax=ax)
    ax.set_xlabel('LSTM correlation')
    ax.set_ylabel('WRTDS correlation')
    ax.set_title('power of seasonal signal')
    for siteNo in siteNoP:
        iS = siteNoLst.index(siteNo)
        circle = plt.Circle([corrMat[iS, ic, 1], corrMat[iS, ic, 2]],
                            0.05, color='red', fill=False)
        ax.add_patch(circle)

    ax = fig.add_subplot(gsM[0, 1:3])
    axplot.mapPoint(ax, lat, lon, corrMat[indS, ic, 1], s=24, cmap='viridis')
    ax.set_title('map of LSTM correlation')
    for siteNo in siteNoP:
        iS = siteNoCode.index(siteNo)
        circle = plt.Circle([lon[iS], lat[iS]], 1,
                            color='red', fill=False)
        ax.add_patch(circle)

    ax = fig.add_subplot(gsM[0, 3:])
    axplot.mapPoint(ax, lat, lon, pMat[indS, ic], s=24, cmap='viridis')
    ax.set_title('map of 1-year power')
    for siteNo in siteNoP:
        iS = siteNoCode.index(siteNo)
        circle = plt.Circle([lon[iS], lat[iS]], 1,
                            color='red', fill=False)
        ax.add_patch(circle)

    for k, siteNo in enumerate(siteNoP):
        v1 = dictLSTM[siteNo][code].values
        v2 = dictWRTDS[siteNo][code].values
        o = dictObs[siteNo][code].values
        t = dictObs[siteNo].index.values
        iS = siteNoLst.index(siteNo)
        # ts
        legLst = ['LSTM {:.2f}'.format(corrMat[iS, ic, 1]),
                  'WRTDS {:.2f}'.format(corrMat[iS, ic, 2]), 'Obs']
        ax = fig.add_subplot(gsM[k+1, :])
        axplot.plotTS(ax, t[indT2], [v1[indT2], v2[indT2], o[indT2]],
                      styLst='--*', cLst='rbk', legLst=legLst)
        # ax.set_title('Water Temp at site {}'.format(siteNo))
        shortName = usgs.codePdf.loc[code]['shortName']
        titleStr = '{} at site {}; power = {:.2f}'.format(
            shortName, siteNo, pMat[iS, ic])
        ax.text(.5, .9, titleStr, horizontalalignment='center',
                transform=ax.transAxes)
    fig.show()
    dirFig = r'C:\Users\geofk\work\paper\waterQuality'
    fig.savefig(os.path.join(dirFig, 'plot_{}'.format(code)))
