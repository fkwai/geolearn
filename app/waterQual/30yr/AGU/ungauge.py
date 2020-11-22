import matplotlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import os
import json


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSiteN5 = json.load(f)
with open(os.path.join(dirSel, 'dictRB_Y30N2.json')) as f:
    dictSiteN2 = json.load(f)
codeLst = sorted(usgs.newC)

dictSite = dict()
for code in usgs.newC+['comb']:
    siteNoCode = list(set(dictSiteN2[code])-set(dictSiteN5['comb']))
    dictSite[code] = siteNoCode
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
dictLSTMLst = list()
# LSTM
labelLst = ['QTFP_C', 'FP_QC']
for label in labelLst:
    dictLSTM = dict()
    trainSet = 'comb-B10'
    outName = '{}-{}-{}-{}-ungauge'.format('rbWN5', 'comb', label, 'comb-B10')
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
dictLSTM1 = dictLSTMLst[0]
dictLSTM2 = dictLSTMLst[1]
corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
rmseMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v0 = dictObs[siteNo][code].iloc[ind2].values
        v1 = dictLSTM1[siteNo][code].iloc[ind2].values
        v2 = dictLSTM2[siteNo][code].iloc[ind2].values
        v3 = dictWRTDS[siteNo][code].iloc[ind2].values
        rmse1, corr1 = utils.stat.calErr(v1, v0)
        rmse2, corr2 = utils.stat.calErr(v2, v0)
        rmse3, corr3 = utils.stat.calErr(v3, v0)
        corrMat[indS, ic, 0] = corr1
        corrMat[indS, ic, 1] = corr2
        corrMat[indS, ic, 2] = corr3
        rmseMat[indS, ic, 0] = rmse1
        rmseMat[indS, ic, 1] = rmse2
        rmseMat[indS, ic, 2] = rmse3

matplotlib.rcParams.update({'font.size': 13})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['LSTM w/o Q', 'LSTM w/ Q', 'WRTDS']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(corrMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='bgr',
                      label2=labLst2, figsize=(20, 5), yRange=[0, 1])
fig.show()

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['LSTM w/o Q', 'LSTM w/ Q', 'WRTDS']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 1, 2]:
        temp.append(rmseMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='bgr',
                      label2=labLst2, figsize=(20, 5), sharey=False)
fig.show()
