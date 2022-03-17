
import scipy
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


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.newC
siteNoLst = DF.siteNoLst

# gageII
dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)
dfG = gageII.removeField(dfG)
colLst = dfG.columns

# load linear
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\Q\param'
matQ = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matQ[:, k] = dfCorr['rsq'].values

# load seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\S\param'
matS = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matS[:, k] = dfCorr['rsq'].values

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
count = np.nansum(matB, axis=0)
matRm = count < 100
matQ[matRm] = np.nan
matS[matRm] = np.nan

# cart
dfA = pd.DataFrame(index=range(10), columns=codeLst)
dfV = pd.DataFrame(index=range(10), columns=codeLst)

for code in codeLst:
    ic = codeLst.index(code)
    matAll = matQ[:, ic]
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
    for k in range(10):
        ind0 = np.arange(len(siteNoCode))
        ind1, ind2, feat, th = subTree(ind0, varLst=colLst)
        dfA.at[k, code] = feat
        dfV.at[k, code] = th
        colLst.remove(feat)
dfA.to_csv('temp2')
dfV.to_csv('temp1')

(unique, counts) = np.unique(dfA.values.flatten(), return_counts=True)
