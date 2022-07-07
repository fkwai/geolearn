from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sklearn.tree
import matplotlib.gridspec as gridspec
from hydroDL.master import basinFull
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin


# DF = dbBasin.DataFrameBasin('G200')
codeLst = DF.varC
siteNoLst = DF.siteNoLst

# LSTM
ep = 500
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)
outFolder = basinFull.nameFolder(outName)
corrName1 = 'corrQ-{}-Ep{}.npy'.format(trainSet, ep)
corrName2 = 'corrQ-{}-Ep{}.npy'.format(testSet, ep)
corrFile1 = os.path.join(outFolder, corrName1)
corrFile2 = os.path.join(outFolder, corrName2)
corrL1 = np.load(corrFile1)
corrL2 = np.load(corrFile2)

# WRTDS
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
corrName1 = 'corr-{}-{}-{}.npy'.format('G200N', trainSet, testSet)
corrName2 = 'corr-{}-{}-{}.npy'.format('G200N', testSet, testSet)
corrFile1 = os.path.join(dirWRTDS, corrName1)
corrFile2 = os.path.join(dirWRTDS, corrName2)
corrW1 = np.load(corrFile1)
corrW2 = np.load(corrFile2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan

# load basin attributes
regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM',
             'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
dfG = gageII.readData(siteNoLst=siteNoLst)
fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
tabT = pd.read_csv(fileT).set_index('PNV_CODE')
for code in range(1, 63):
    siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
    dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
dfG = gageII.updateCode(dfG)
dfG = gageII.removeField(dfG)
#
dfA = pd.DataFrame(index=range(10), columns=codeLst)
dfV = pd.DataFrame(index=range(10), columns=codeLst)

for code in codeLst:
    ic = codeLst.index(code)
    matAll = corrL2[:, ic]**2-corrW2[:, ic]**2
    [mat], indS = utils.rmNan([matAll])
    siteNoCode = [siteNoLst[ind] for ind in indS]
    dfGC = dfG.loc[siteNoCode]

    def subTree(indInput, varLst):
        x = dfGC.iloc[indInput][varLst].values.astype(float)
        y = mat[indInput]
        tt = 0.1
        y[y <= -tt] = 0
        y[y >= tt] = 2
        y[(y < tt) & (y > -tt)] = 1
        x[np.isnan(x)] = -99
        clf = sklearn.tree.DecisionTreeClassifier(
            max_depth=1, min_samples_leaf=0.2)
        clf = clf.fit(x, y)
        tree = clf.tree_
        feat = varLst[tree.feature[0]]
        th = tree.threshold[0]
        indLeft = np.where(x[:, tree.feature[0]] <= tree.threshold[0])[0]
        indRight = np.where(x[:, tree.feature[0]] > tree.threshold[0])[0]
        indLeftG = indInput[indLeft]
        indRightG = indInput[indRight]
        return indLeftG, indRightG, feat, th

    colLst = dfGC.columns.tolist()
    # for yr in range(1950, 2010):
    #     colLst.remove('PPT{}_AVG'.format(yr))
    #     colLst.remove('TMP{}_AVG'.format(yr))
    # for yr in range(1900, 2010):
    #     colLst.remove('wy{}'.format(yr))
    # monthLst = ['JAN', 'FEB', 'APR', 'MAY', 'JUN',
    #             'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    # for m in monthLst:
    #     colLst.remove('{}_PPT7100_CM'.format(m))
    #     colLst.remove('{}_TMP7100_DEGC'.format(m))
    for k in range(10):
        ind0 = np.arange(len(siteNoCode))
        ind1, ind2, feat, th = subTree(ind0, varLst=colLst)
        dfA.at[k, code] = feat
        dfV.at[k, code] = th
        colLst.remove(feat)
dfA.to_csv('temp2')
dfV.to_csv('temp1')

(unique, counts) = np.unique(dfA.values.flatten(), return_counts=True)
