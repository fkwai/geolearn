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
        v0 = dictObs[siteNo][code].iloc[ind2].values
        v1 = dictLSTM[siteNo][code].iloc[ind2].values
        v2 = dictWRTDS[siteNo][code].iloc[ind2].values
        rmse1, corr1 = utils.stat.calErr(v1, v0)
        rmse2, corr2 = utils.stat.calErr(v2, v0)
        rmse3, corr3 = utils.stat.calErr(v1, v2)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3

dfG = gageII.readData(
    varLst=None, siteNoLst=siteNoLst)
varG = 'DDENS_2009'


# plot 121
importlib.reload(axplot)
codeLst2 = ['00095', '00400', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
fig, axes = plt.subplots(5, 4)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 5, 4)
    ax = axes[j, i]
    ic = codeLst.index(code)
    # x = corrMat[:, ic, 1]
    x = dfG[varG].values
    y = corrMat[:, ic, 2]
    # c = np.argsort(countMat2[:, ind])
    ax.plot(x, y, '*')
    # rmse, corr = utils.stat.calErr(x, y)
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
    # print(i, j)
    # _ = ax.set_aspect('equal')
# plt.subplots_adjust(wspace=0, hspace=0)
# fig.colorbar()
fig.show()
