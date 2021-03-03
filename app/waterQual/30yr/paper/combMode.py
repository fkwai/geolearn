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
if True:
    # LSTM
    label = 'QTFP_C'
    # comb
    dictComb = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = basins.loadSeq(outName, siteNo)
        dictComb[siteNo] = df
    # comb
    dictSolo = dict()
    for code in codeLst:
        dictCode = dict()
        trainSet = '{}-B10'.format(code)
        outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        for k, siteNo in enumerate(siteNoLst):
            print('\t {} site {}/{}'.format(code, k, len(siteNoLst)), end='\r')
            df = basins.loadSeq(outName, siteNo)
            dictCode[siteNo] = df
        dictSolo[code] = dictCode
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
    corrMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
    rmseMat = np.full([len(siteNoLst), len(codeLst), 4], np.nan)
    for ic, code in enumerate(codeLst):
        for k, ind in enumerate([ind1, ind2]):
            for siteNo in dictSite[code]:
                indS = siteNoLst.index(siteNo)
                v1 = dictComb[siteNo][code].iloc[ind].values
                v2 = dictSolo[code][siteNo][code].iloc[ind].values
                v3 = dictObs[siteNo][code].iloc[ind].values
                vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
                rmse1, corr1 = utils.stat.calErr(vv1, vv3)
                rmse2, corr2 = utils.stat.calErr(vv2, vv3)
                corrMat[indS, ic, k*2] = corr1
                corrMat[indS, ic, k*2+1] = corr2


# significance test
dfS = pd.DataFrame(index=codeLst, columns=['rmse', 'corr'])
for k, code in enumerate(codeLst):
    a = corrMat[:, k, 2]
    b = corrMat[:, k, 3]
    aa, bb = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'corr'] = p


# plot box corr
labLst1 = list()
codePlot = sorted(usgs.newC)
codePlot.remove('00950')
for code in codePlot:
    lab = '{}\n{}'.format(
        usgs.codePdf.loc[code]['shortName'], code)
    labLst1.append(lab)
labLst2 = ['train solo', 'train comb', 'test solo', 'test comb']
dataBox = list()
for code in codePlot:
    temp = list()
    for i in [1, 0, 3, 2]:
        temp.append(corrMat[:, codeLst.index(code), i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='mcrb',
                      label2=labLst2, figsize=(16, 5), yRange=[0, 1])
fig.show()
dirFig = r'C:\Users\geofk\work\paper\waterQuality'
fig.savefig(os.path.join(dirFig, 'box_comb'))
