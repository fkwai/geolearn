
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
dictLSTMLst = list()
# LSTM
labelLst = ['QTFP_C']
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
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10', 'output')
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
    df = waterQuality.readSiteTS(siteNo, varLst=codeLst, freq='W')
    dictObs[siteNo] = df

# calculate correlation
tt = np.datetime64('2010-01-01')
ind1 = np.where(df.index.values < tt)[0]
ind2 = np.where(df.index.values >= tt)[0]
dictLSTM = dictLSTMLst[0]
corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictLSTM[siteNo][code].iloc[ind2].values
        v2 = dictWRTDS[siteNo][code].iloc[ind2].values
        v3 = dictObs[siteNo][code].iloc[ind2].values
        rmse1, corr1 = utils.stat.calErr(v1, v2)
        rmse2, corr2 = utils.stat.calErr(v1, v3)
        rmse3, corr3 = utils.stat.calErr(v2, v3)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3

# plot ts
code = '00955'
iCode = codeLst.index(code)
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
siteNoLstCode = dictSite[code]
matMapLst = [corrMat[indS, iCode, 0],
             corrMat[indS, iCode, 1], corrMat[indS, iCode, 2]]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']


def funcMap():
    figM, axM = plt.subplots(1, 3, figsize=(12, 4))
    for k in range(3):
        axplot.mapPoint(axM[k], lat, lon, matMapLst[k], vRange=[0, 1], s=16)
    axM[0].set_title('corr(LSTM, WRTDS)')
    axM[1].set_title('corr(LSTM, obs)')
    axM[2].set_title('corr(WRTDS, obs)')
    figP = plt.figure(figsize=[16, 6])
    gs = gridspec.GridSpec(3, 12)
    axTS = figP.add_subplot(gs[0, :])
    axH1 = figP.add_subplot(gs[1, :4])
    axH2 = figP.add_subplot(gs[1, 4:8])
    axH3 = figP.add_subplot(gs[1, 8:])
    axP1 = figP.add_subplot(gs[2, :6])
    axP2 = figP.add_subplot(gs[2, 6:])
    axP = np.array([axTS, axH1, axH2, axH3, axP1, axP2])
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    [axTS, axH1, axH2, axH3, axP1, axP2] = axP
    siteNo = siteNoLstCode[iP]
    outName1 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QTFP_C', trainSet)
    outName2 = '{}-{}-{}-{}'.format(dataName, 'comb', 'QT_C', trainSet)
    dfL1 = basins.loadSeq(outName1, siteNo)
    dfL2 = basins.loadSeq(outName2, siteNo)
    dfW = pd.read_csv(os.path.join(dirWRTDS, siteNo),
                      index_col=None).set_index('date')
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq='W')
    dfOD = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq='D')
    t = dfO.index
    # ts
    tBar = np.datetime64('2010-01-01')
    sd = np.datetime64('1980-01-01')
    legLst = ['LSTM QTFP', 'LSTM QT', 'WRTDS', 'Obs']
    axplot.plotTS(axTS, t, [dfL1[code], dfL2[code], dfW[code], dfO[code]],
                  tBar=tBar, sd=sd, styLst='---*', cLst='mrbk', legLst=legLst)
    corrL = corrMat[indS[iP], iCode, 0]
    corrW = corrMat[indS[iP], iCode, 1]
    axplot.titleInner(
        axTS, 'siteNo {} {:.2f} {:.2f}'.format(siteNo, corrL, corrW))
    axTS.legend()
    # hist
    axH1.hist(dfOD[code].values, density=True, bins=50)
    axplot.titleInner(axH1, 'histogram {}'.format(shortName))
    axH2.hist(dfOD['00060'].values, density=True, bins=50)
    axplot.titleInner(axH2, 'histogram {}'.format('Q'))
    axH3.hist(np.log(dfOD['00060'].values+1), density=True, bins=50)
    axplot.titleInner(axH3, 'histogram {}'.format('log Q'))
    # periodgram
    freqQ, powerQ, pQ = calPower('00060', dfOD)
    freqC, powerC, pC = calPower(code, dfOD)
    axP1.plot(1/freqQ, powerC, '-*b', label='Periodograms')
    axP1.plot(1/freqQ, pQ, '-*r', label='baluev probability')
    axplot.titleInner(axP1, 'streamflow')
    axP1.legend()
    axP2.plot(1/freqC, powerC, '-*b', label='Periodograms')
    axP2.plot(1/freqC, pC, '-*r', label='baluev probability')
    axplot.titleInner(axP2, shortName)
    axP2.legend()


def calPower(code, df):
    tt = df.index.values
    dfD = df[df[code].notna().values]
    t = dfD.index.values
    x = (t.astype('datetime64[D]') -
         np.datetime64('1979-01-01')).astype(np.float)
    y = dfD[code].values
    nt = len(tt)
    freq = np.fft.fftfreq(nt)[1:]
    ind = np.where((1/freq >= 0) & (1/freq < 1000))[0]
    freq = freq[ind]
    ls = LombScargle(x, y)
    power = ls.power(freq)
    p = ls.false_alarm_probability(power)
    return freq, power, 1-p


importlib.reload(axplot)
figM, figP = figplot.clickMap(funcMap, funcPoint)
