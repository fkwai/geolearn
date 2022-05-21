

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree

codeLst = usgs.varC

DF = dbBasin.DataFrameBasin('G200')
# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])).astype(int).astype(float)
count = np.nansum(matB, axis=0)

matRm = count < 200

# load linear/seasonal
dirP = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'
labLst = ['Q', 'S', 'QS']
dictS = dict()
for lab in labLst:
    dirS = dirP.format(lab)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirS, code)
        dfCorr = pd.read_csv(
            filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictS[lab] = matLR

matQ = dictS['Q']
matS = dictS['S']
matQS = dictS['QS']

#  gageII
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
dfG = gageII.updateCode(dfG)
# remove some attrs
colLst = dfG.columns.tolist()
for yr in range(1950, 2010):
    colLst.remove('PPT{}_AVG'.format(yr))
    colLst.remove('TMP{}_AVG'.format(yr))
for yr in range(1900, 2010):
    colLst.remove('wy{}'.format(yr))
monthLst = ['JAN', 'FEB', 'APR', 'MAY', 'JUN',
            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
for m in monthLst:
    colLst.remove('{}_PPT7100_CM'.format(m))
    colLst.remove('{}_TMP7100_DEGC'.format(m))


def subTree(x, y):
    x[np.isnan(x)] = -99
    clf = sklearn.tree.DecisionTreeRegressor(max_depth=1)
    clf = clf.fit(x, y)
    tree = clf.tree_
    feat = tree.feature[0]
    th = tree.threshold[0]
    indLeft = np.where(x[:, tree.feature[0]] <= tree.threshold[0])[0]
    indRight = np.where(x[:, tree.feature[0]] > tree.threshold[0])[0]
    return indLeft, indRight, feat, th


dfA = pd.DataFrame(index=range(10), columns=codeLst)
dfV = pd.DataFrame(index=range(10), columns=codeLst)


# for a code
for k, code in enumerate(codeLst):
    [mat], indS = utils.rmNan([matQS[:, k]])
    siteNoCode = [DF.siteNoLst[ind] for ind in indS]
    dfGC = dfG.loc[siteNoCode]
    colTemp = colLst.copy()
    for k in range(10):
        ind0 = np.arange(len(siteNoCode))
        x = dfGC[colTemp].values
        y = mat
        ind1, ind2, feat, th = subTree(x, y)
        dfA.at[k, code] = colTemp[feat]
        dfV.at[k, code] = th
        colTemp.remove(colTemp[feat])

saveDir = r'C:\Users\geofk\work\waterQuality\paper\G200\simplicity'
dfA.to_csv(os.path.join(saveDir,'top10attr.csv'))
dfV.to_csv(os.path.join(saveDir,'top10values.csv'))
