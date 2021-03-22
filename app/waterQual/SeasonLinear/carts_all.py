from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sklearn.tree
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
    outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
    dictLSTM, dictWRTDS, dictObs = wq.loadModel(
        siteNoLst, outNameLSTM, codeLst)
    corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)
    # load basin attributes
    dfG = gageII.readData(siteNoLst=siteNoLst)
    dfG = gageII.updateRegion(dfG)
    dfG = gageII.updateCode(dfG)

# fileGlim = os.path.join(kPath.dirData, 'USGS', 'GLiM', 'tab_1KM')
# tabGlim = pd.read_csv(fileGlim, dtype={'siteNo': str}).set_index('siteNo')
# dfG = tabGlim


#
dfA = pd.DataFrame(index=range(10), columns=codeLst)
dfV = pd.DataFrame(index=range(10), columns=codeLst)

for code in codeLst: 
    ic = codeLst.index(code)
    matAll = corrMat[:, ic, 1]**2-corrMat[:, ic, 2]**2
    [mat], indS = utils.rmNan([matAll])
    siteNoCode = [siteNoLst[ind] for ind in indS]
    dfGC = dfG.loc[siteNoCode]

    def subTree(indInput, varLst):
        x = dfGC.iloc[indInput][varLst].values.astype(float)
        y = mat[indInput]
        x[np.isnan(x)] = -99
        clf = sklearn.tree.DecisionTreeRegressor(max_depth=1,min_samples_leaf=0.2)
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
