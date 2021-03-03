
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
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
dictLSTMLst = list()
# LSTM
labelLst = ['QT_C', 'QTFP_C', 'FP_QC']
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
dictLSTM = dictLSTMLst[1]
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

# bSite
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
bSiteMat = np.full([len(siteNoLst), len(codeLst)], False)
for k, code in enumerate(codeLst):
    siteNoCode = dictSite[code]
    indS = [siteNoLst.index(siteNo) for siteNo in siteNoCode]
    bSiteMat[indS, k] = True


# load regions
regionLst = [
    'ECO2_BAS_DOM',
    'NUTR_BAS_DOM',
    'HLR_BAS_DOM_100M',
    'PNV_BAS_DOM',
]
dfG = gageII.readData(varLst=regionLst+['LAT_GAGE', 'LNG_GAGE', 'CLASS'])
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']

# define what to show
v = dfG['PNV_BAS_DOM2'].values
vLst = [0, 1]
vStrLst = ['Non-ref', 'ref']
labLst2 = list()
for s1 in vStrLst:
    for s2 in ['LSTM', 'WRTDS']:
        labLst2.append('{} {}'.format(s1, s2))

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
dataBox = list()
for k, code in enumerate(codeLst):
    bSite = bSiteMat[:, k]
    temp = list()
    for vv in vLst:
        for i in [1, 2]:
            bv = v == vv
            indS = np.where(bSite & bv)[0]
            temp.append(corrMat[indS, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
fig.show()
