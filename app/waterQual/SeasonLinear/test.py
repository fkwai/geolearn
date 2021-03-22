
import importlib
import numpy as np
import os
import pandas as pd
import json
from hydroDL.master import basins
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import usgs, gageII
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot


dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['comb']
codeLst = sorted(usgs.newC)

# load Linear and Seasonal model
dictL = dict()
dirL = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-WL', 'All', 'output')
dictS = dict()
dirS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-WS', 'All', 'output')
for dirTemp, dictTemp in zip([dirL, dirS], [dictL, dictS]):
    for k, siteNo in enumerate(siteNoLst):
        print('\t WRTDS site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirTemp, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        dictTemp[siteNo] = df

dictObs = dict()
for k, siteNo in enumerate(siteNoLst):
    print('\t USGS site {}/{}'.format(k, len(siteNoLst)), end='\r')
    df = waterQuality.readSiteTS(
        siteNo, varLst=['00060']+codeLst, freq='W', rmFlag=True)
    dictObs[siteNo] = df

# calculate correlation
corrMatTemp = np.full([len(siteNoLst), len(codeLst), 2], np.nan)
for ic, code in enumerate(codeLst):
    for siteNo in dictSite[code]:
        indS = siteNoLst.index(siteNo)
        v1 = dictL[siteNo][code].values
        v2 = dictS[siteNo][code].values
        v0 = dictObs[siteNo][code].values
        (vv0, vv1, vv2), indV = utils.rmNan([v0, v1, v2])
        rmse1, corr1 = utils.stat.calErr(vv1, vv0)
        rmse2, corr2 = utils.stat.calErr(vv2, vv0)
        corrMatTemp[indS, ic, 0] = corr1
        corrMatTemp[indS, ic, 1] = corr2

rMat = corrMatTemp**2
codeLst2 = ['00915', '00925', '00930', '00935', '00940', '00945',
            '00955', '70303', '80154']
[nfy, nfx] = [3, 3]

codeLst2 = ['00010', '00300', '00405', '00600', '00605',
            '00618', '00660', '00665', '00681', '00915',
            '00925', '00930', '00935', '00940', '00945',
            '00950', '00955', '70303', '71846', '80154']
nfy, nfx = [4, 5]

fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = rMat[:, ic, 1]
    y = rMat[:, ic, 0]
    axplot.plot121(ax, x, y, vR=[0, 1])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateRegion(dfG)
dfG = gageII.updateCode(dfG)
cMat=
fig, axes = plt.subplots(nfy, nfx)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, nfy, nfx)
    ax = axes[j, i]
    ic = codeLst.index(code)
    x = rMat[:, ic, 1]
    y = rMat[:, ic, 0]
    axplot.scatter121(ax, x, y, cvR=[0, 1])
    titleStr = '{} {} '.format(
        code, usgs.codePdf.loc[code]['shortName'])
    axplot.titleInner(ax, titleStr)
fig.show()

# Cart
dfA = pd.DataFrame(index=range(10), columns=codeLst)
dfV = pd.DataFrame(index=range(10), columns=codeLst)

for code in codeLst:
    ic = codeLst.index(code)
    matAll = rMat[]
    [mat], indS = utils.rmNan([matAll])
    siteNoCode = [siteNoLst[ind] for ind in indS]
    dfGC = dfG.loc[siteNoCode]

    def subTree(indInput, varLst):
        x = dfGC.iloc[indInput][varLst].values.astype(float)
        y = mat[indInput]
        x[np.isnan(x)] = -99
        clf = sklearn.tree.DecisionTreeRegressor(
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
