
from astropy.timeseries import LombScargle
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

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
dataName = 'rbWN5'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)
corrMat = np.full([nSite, len(codeLst), 2], np.nan)

# LSTM
label = 'QTFP_C'
trainSet = 'comb-B10'
testSet = 'comb-A10'
outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
master = basins.loadMaster(outName)
subset = testSet
yP, ycP = basins.testModel(
    outName, subset, wqData=wqData, ep=ep, reTest=reTest)
ind = wqData.subset[subset]
info = wqData.info.iloc[ind].reset_index()
for iCode, code in enumerate(codeLst):
    ic = wqData.varC.index(code)
    if len(wqData.c.shape) == 3:
        p = yP[-1, :, master['varY'].index(code)]
        o = wqData.c[-1, ind, ic]
    elif len(wqData.c.shape) == 2:
        p = ycP[:, master['varYC'].index(code)]
        o = wqData.c[ind, ic]
    for siteNo in dictSite[code]:
        iS = siteNoLst.index(siteNo)
        indS = info[info['siteNo'] == siteNo].index.values
        rmse, corr = utils.stat.calErr(p[indS], o[indS])
        corrMat[iS, iCode, 0] = corr
        # rmseMat[iS, iCode, iT*2] = rmse

# WRTDS
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-W', 'B10')
file2 = os.path.join(dirWrtds, '{}-{}-corr'.format('B10N5', 'A10N5'))
dfCorr2 = pd.read_csv(file2, dtype={'siteNo': str}).set_index('siteNo')
for iCode, code in enumerate(codeLst):
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    corrMat[indS, iCode, 1] = dfCorr2.iloc[indS][code].values

# plot ts
code = '00915'
iCode = codeLst.index(code)
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
siteNoLstCode = dictSite[code]
matMap = corrMat[indS, iCode, 0]-corrMat[indS, iCode, 1]
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstCode)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
shortName = usgs.codePdf.loc[code]['shortName']


def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(12, 4))
    axplot.mapPoint(axM, lat, lon, matMap, vRange=[-0.3, 0.3], s=16)
    axM.set_title('testing corr LSTM - corr WRTDS')
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
    dfW = pd.read_csv(os.path.join(dirWrtds, 'output', siteNo),
                      index_col=None).set_index('date')
    dfO = waterQuality.readSiteTS(siteNo, codeLst+['00060'], freq=wqData.freq)
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
