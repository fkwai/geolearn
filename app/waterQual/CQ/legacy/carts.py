import sklearn.tree
import os
import pandas as pd
import numpy as np
from hydroDL import kPath
from hydroDL.data import usgs, gageII
from hydroDL.post import axplot
import matplotlib.pyplot as plt

dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
dfS = pd.read_csv(os.path.join(dirCQ, 'slope'), dtype={
    'siteNo': str}).set_index('siteNo')
dfN = pd.read_csv(os.path.join(dirCQ, 'nSample'), dtype={
                  'siteNo': str}).set_index('siteNo')
siteNoLst = dfS.index.tolist()
codeLst = dfS.columns.tolist()

dropColLst = ['STANAME', 'WR_REPORT_REMARKS',
              'ADR_CITATION', 'SCREENING_COMMENTS']
dfX = gageII.readData(siteNoLst=siteNoLst).drop(columns=dropColLst)
dfX = gageII.updateCode(dfX)
dfCrd = gageII.readData(varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)

code = '00955'
indValid = np.where((~np.isnan(dfS['00955'].values))
                    & (dfN['00955'].values > 10))[0]
dataAll = dfS[code][indValid]
vr = np.max([np.abs(np.percentile(dataAll, 1)),
             np.abs(np.percentile(dataAll, 99))])
vRange = [-vr, vr]


def subTree(indInput):
    x = dfX.values[indInput, :]
    y = dfS[code].values[indInput]
    x[np.isnan(x)] = -99
    clf = sklearn.tree.DecisionTreeRegressor(max_depth=1)
    clf = clf.fit(x, y)
    tree = clf.tree_
    feat = dfX.columns[tree.feature[0]]
    th = tree.threshold[0]
    indLeft = np.where(x[:, tree.feature[0]] <= tree.threshold[0])[0]
    indRight = np.where(x[:, tree.feature[0]] > tree.threshold[0])[0]
    indLeftG = indInput[indLeft]
    indRightG = indInput[indRight]
    return indLeftG, indRightG, feat, th


def plotCdf(ax, indInput, indLeft, indRight):
    cLst = 'gbr'
    labLst = ['parent', 'left', 'right']
    y0 = dfS[code].values[indInput]
    y1 = dfS[code].values[indLeft]
    y2 = dfS[code].values[indRight]
    dataLst = [y0, y1, y2]
    for k, data in enumerate(dataLst):
        xSort = np.sort(data[~np.isnan(data)])
        yRank = np.arange(1, len(xSort)+1) / float(len(xSort))
        ax.plot(xSort, yRank, color=cLst[k], label=labLst[k])
        ax.set_xlim(vRange)
    ax.legend(loc='best', frameon=False)


def plotMap(ax, indInput):
    lat = dfCrd['LAT_GAGE'][indInput]
    lon = dfCrd['LNG_GAGE'][indInput]
    data = dfS[code][indInput]
    axplot.mapPoint(ax, lat, lon, data, vRange=vRange, s=10)


indInput = indValid
indLeft, indRight, feat, th = subTree(indInput)
fig, ax = plt.subplots(1, 1)
plotCdf(ax, indInput, indLeft, indRight)
fig.show()

fig, axes = plt.subplots(2, 1)
plotMap(axes[0], indLeft)
plotMap(axes[1], indRight)
fig.show()
