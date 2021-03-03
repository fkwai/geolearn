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
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

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
    dictLSTMLst = list()
    # LSTM
    labelLst = ['FP_QC']
    for label in labelLst:
        dictLSTM = dict()
        trainSet = 'comb-B10'
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        for k, siteNo in enumerate(siteNoLst):
            print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
            df = basins.loadSeq(outName, siteNo)
            dictLSTM[siteNo] = df
        dictLSTMLst.append(dictLSTM)
    # WRTDS
    dictWRTDS = dict()
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-W', 'B10', 'output')
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
        df = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+codeLst, freq='W')
        dictObs[siteNo] = df

    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    ind1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
    ind2 = np.where(df.index.values >= tt)[0]
    dictLSTM = dictLSTMLst[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictWRTDS[siteNo][code].iloc[ind2].values
            v3 = dictObs[siteNo][code].iloc[ind2].values
            vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
            rmse1, corr1 = utils.stat.calErr(vv1, vv2)
            rmse2, corr2 = utils.stat.calErr(vv1, vv3)
            rmse3, corr3 = utils.stat.calErr(vv2, vv3)
            corrMat[indS, ic, 0] = corr1
            corrMat[indS, ic, 1] = corr2
            corrMat[indS, ic, 2] = corr3

# color mat - region
regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM',
             'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
dfG = gageII.readData(siteNoLst=siteNoLst)
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
dfG = gageII.updateCode(dfG)
cVar = 'NWIS_DRAIN_SQKM'
cMat = dfG[cVar].values
# cR = [np.nanpercentile(cMat, 10), np.nanpercentile(cMat, 90)]
cR = [np.nanmin(cMat), np.nanmax(cMat)]


# plot 121
plt.close('all')
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']

# attr vs diff
fig, axes = plt.subplots(5, 4)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = cMat
    y = corrMat[:, ic, 1]**2-corrMat[:, ic, 2]**2
    ax.plot(x, y, '*')
    ax.plot([np.nanmin(x), np.nanmax(x)], [0, 0], 'k-')
    ax.set_ylim([-0.3, 0.3])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

gsM = gridspec.GridSpec(50, 41)
figM = plt.figure(figsize=[10, 10])
ticks = [-0.5, 0, 0.5, 1]
axM = list()
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    axM.append(figM.add_subplot(gsM[(j)*10:(j+1)*10, i*10:(i+1)*10]))

for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axM[k]
    ic = codeLst.index(code)
    x = corrMat[:, ic, 1]
    y = corrMat[:, ic, 2]
    c = cMat
    # c = np.argsort(countMat2[:, ind])
    sc = axplot.scatter121(ax, x, y, c, vR=cR)
    rmse, corr = utils.stat.calErr(x, y)
    titleStr = '{} {} {:.2f}'.format(
        code, usgs.codePdf.loc[code]['shortName'], corr)
    _ = ax.set_xlim([ticks[0], ticks[-1]])
    _ = ax.set_ylim([ticks[0], ticks[-1]])
    _ = ax.set_yticks(ticks)
    _ = ax.set_xticks(ticks)
    axplot.titleInner(ax, titleStr)
    ax.set_label(code)
    # print(i, j)
    if i != 0:
        _ = ax.set_yticklabels([])
    if j != 4:
        _ = ax.set_xticklabels([])
    # _ = ax.set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
cax = figM.add_subplot(gsM[:, 40])
figM.colorbar(sc, cax=cax)
figM.suptitle('corr of LSTM vs WRTDS colored by {}'.format(cVar))


figP = plt.figure(figsize=[12, 6])
gsP = gridspec.GridSpec(3, 4)
axP = list()
axP.append(figP.add_subplot(gsP[0, 0]))
axP.append(figP.add_subplot(gsP[0, 1]))
axP.append(figP.add_subplot(gsP[0, 2]))
axP.append(figP.add_subplot(gsP[1, :2]))
axP.append(figP.add_subplot(gsP[2, :2]))
axP.append(figP.add_subplot(gsP[1:, 2]))


def plotP(xx, yy, cc, iP, code):
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    # maps
    axplot.mapPoint(axP[0], lat, lon, xx, vRange=[-0.5, 1], s=16)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[0].add_patch(circle)
    axplot.mapPoint(axP[1], lat, lon, yy, vRange=[-0.5, 1], s=16)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[1].add_patch(circle)
    axplot.mapPoint(axP[2], lat, lon, cc, vRange=cR, s=16)
    circle = plt.Circle([lon[iP], lat[iP]], 2, color='black', fill=False)
    axP[2].add_patch(circle)
    siteNo = siteNoLst[iP]
    # ts
    v0 = dictObs[siteNo][code].values
    v1 = dictLSTM[siteNo][code].values
    v2 = dictWRTDS[siteNo][code].values
    t = dictObs[siteNo].index.values
    legLst = ['LSTM', 'WRTDS', 'Obs']
    axplot.plotTS(axP[3], t[ind1], [v1[ind1], v2[ind1], v0[ind1]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    axplot.plotTS(axP[4], t[ind2], [v1[ind2], v2[ind2], v0[ind2]],
                  styLst='--*', cLst='rbk', legLst=legLst)
    # cq
    q = dictObs[siteNo]['00060'].values
    c = dictObs[siteNo][code].values
    td = dictObs[siteNo].index.dayofyear
    sc = axP[5].scatter(np.log(q), c, c=td, cmap='hsv', vmin=0, vmax=365)
    # figP.colorbar(sc, ax=axP[5])
    figP.suptitle('code {} {}; siteNo {} \n corrLSTM {:.2f}; corrWRTDS {:.2f}; {} {}'.format(
        code, usgs.codePdf.loc[code]['shortName'], siteNo, xx[iP], yy[iP], cVar, cc[iP]))
    figP.show()


def onclick(event):
    xClick = event.xdata
    yClick = event.ydata
    code = event.inaxes.get_label()
    ic = codeLst.index(code)
    xx = corrMat[:, ic, 1]
    yy = corrMat[:, ic, 2]
    cc = cMat.copy()
    cc[np.isnan(xx)] = np.nan
    iP = np.nanargmin(np.sqrt((xClick - xx)**2 + (yClick - yy)**2))
    print(code, xClick, yClick)
    for ax, codeT in zip(axM, codeLst2):
        [p.remove() for p in reversed(ax.patches)]
        xc = corrMat[iP, codeLst.index(codeT), 1]
        yc = corrMat[iP, codeLst.index(codeT), 2]
        color = 'red' if codeT == code else 'black'
        circle = plt.Circle([xc, yc], 0.1, color=color, fill=False)
        ax.add_patch(circle)
    for ax in axP:
        ax.clear()
    plotP(xx, yy, cc, iP, code)
    figM.canvas.draw()
    figP.canvas.draw()


figM.canvas.mpl_connect('button_press_event',
                        lambda event: onclick(event))
figM.show()
figP.show()

# plt.close('all')
