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

# count
fileSiteNo = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeCount = sorted(usgs.codeLst)
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
countMatAll = np.load(os.path.join(dirInv, 'matCountWeekly.npy'))

countMat = np.ndarray([len(siteNoLstAll), len(codeCount)])
for ic, code in enumerate(codeCount):
    countMat[:, ic] = np.sum(countMatAll[:, :, ic], axis=1)

# select site
n = 40*2
codeLst = ['00915']
nc = len(codeLst)
icLst = [codeCount.index(code) for code in codeLst]
bMat = countMat[:, icLst] > n
# indSel = np.where(bMat.any(axis=1))
indSel = np.where(bMat.all(axis=1))[0]
siteNoLst = [siteNoLstAll[ind] for ind in indSel]

# WRTDS and gageII
dirWrtds = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-D', 'All')
fileCorr = os.path.join(dirWrtds, 'corr')
dfCorr = pd.read_csv(fileCorr, dtype={'siteNo': str}).set_index('siteNo')
mat = dfCorr.loc[siteNoLst][codeLst].values
dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateCode(dfG)


def subTree(indInput, varLst):
    x = dfG.iloc[indInput][varLst].values
    y = mat[indInput]
    x[np.isnan(x)] = -99
    clf = sklearn.tree.DecisionTreeRegressor(max_depth=1)
    clf = clf.fit(x, y)
    tree = clf.tree_
    feat = varLst[tree.feature[0]]
    th = tree.threshold[0]
    indLeft = np.where(x[:, tree.feature[0]] <= tree.threshold[0])[0]
    indRight = np.where(x[:, tree.feature[0]] > tree.threshold[0])[0]
    indLeftG = indInput[indLeft]
    indRightG = indInput[indRight]
    return indLeftG, indRightG, feat, th


def plotCdf(ax, indInput, indLeft, indRight):
    cLst = 'gbr'
    labLst = ['parent', 'left', 'right']
    y0 = mat[indInput]
    y1 = mat[indLeft]
    y2 = mat[indRight]
    dataLst = [y0, y1, y2]
    for k, data in enumerate(dataLst):
        xSort = np.sort(data[~np.isnan(data)])
        yRank = np.arange(1, len(xSort)+1) / float(len(xSort))
        ax.plot(xSort, yRank, color=cLst[k], label=labLst[k])
        ax.set_xlim([0, 1])
    ax.legend(loc='best', frameon=False)


def plotMap(ax, indInput):
    lat = dfG['LAT_GAGE'][indInput]
    lon = dfG['LNG_GAGE'][indInput]
    data = mat[indInput]
    axplot.mapPoint(ax, lat, lon, data, vRange=[0, 1], s=10)


def divide(indInput, colLst):
    gs = gridspec.GridSpec(2, 2)
    indLeft, indRight, feat, th = subTree(indInput, colLst)
    fig = plt.figure(figsize=[8, 4])
    ax1 = fig.add_subplot(gs[0:2, 0])
    plotCdf(ax1, indInput, indLeft, indRight)
    ax2 = fig.add_subplot(gs[0, 1])
    plotMap(ax2, indLeft)
    ax3 = fig.add_subplot(gs[1, 1])
    plotMap(ax3, indRight)
    fig.suptitle('{} {:.3f}'.format(feat, th))
    fig.show()
    return indLeft, indRight, feat, th


# # node 0
colLst = dfG.columns.tolist()
ind0 = np.arange(len(siteNoLst))
ind1, ind2, feat1, th = divide(ind0, colLst=colLst)
ind3, ind4, feat2, th = divide(ind1, colLst=colLst)
ind5, ind6, feat3, th = divide(ind2, colLst=colLst)

# remove some attrs
colLst = dfG.columns.tolist()
colLst.remove('NO200AVE')
colLst.remove('KFACT_UP')


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
ind1, ind2, feat1, th = divide(ind0, colLst=colLst)
ind3, ind4, feat2, th = divide(ind1, colLst=colLst)
ind5, ind6, feat3, th = divide(ind2, colLst=colLst)
