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
from scipy import stats
from sklearn import decomposition


# load WRTDS results
dirRoot1 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly')
dirRoot2 = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS_weekly_rmq')

code = '00955'
dfRes1 = pd.read_csv(os.path.join(dirRoot1, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')
dfRes2 = pd.read_csv(os.path.join(dirRoot2, 'result', code), dtype={
    'siteNo': str}).set_index('siteNo')

# dfRes1[dfRes1 == -9999] = np.nan
dfGeo = gageII.readData(siteNoLst=dfRes1.index.tolist())
dfGeo = gageII.updateCode(dfGeo)

# select sites
nS = 100
dfR1 = dfRes1[dfRes1['count'] > nS]
siteNoLst = dfR1.index.tolist()
dfR2 = dfRes2.loc[siteNoLst]
dfG = dfGeo.loc[siteNoLst]
dfGN = (dfG-dfG.min())/(dfG.max()-dfG.min())

x = dfGN.values
x[np.isnan(x)] = -1
pca = decomposition.PCA(n_components=10)
pca.fit(x)
xx = pca.transform(x)

dfPCA = pd.DataFrame(index=dfG.columns, columns=[
                     'PCA{}'.format(x) for x in range(10)], data=pca.components_.T)
dfPCA.to_csv('temp')
pca.explained_variance_ratio_
pca.components_[0, :]

# 121
for k in range(10):
    fig, ax = plt.subplots(1, 1)
    ax.plot(xx[:, k], dfR1['corr'].values, '*')
    ax.set_xlabel('PCA{}'.format(k))
    ax.set_ylabel('WRTDS corr')
    fig.show()
