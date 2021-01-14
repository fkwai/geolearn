import matplotlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality, cart
from hydroDL.master import basins
from hydroDL.data import usgs, gageII
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import sklearn.tree
import matplotlib.gridspec as gridspec

from sklearn import decomposition

# load gageII
dfGeo = gageII.readData()
dfGeo = gageII.updateCode(dfGeo)
dfGeo = gageII.removeField(dfGeo)
dirTree = r'C:\Users\geofk\work\waterQuality\C-Q\tree'

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
saveFolder = os.path.join(dirTree, 'tree_pcaG_00915')
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
dfG = dfGeo.loc[siteNoLst]

# pca by group
dictVar = gageII.getVariableDict()
grpLst = list(dictVar.keys())
# grpLst = ['LC06_Basin', 'LC06_Mains100', 'LC_Crops']
matPcaLst = list()
pcaNameLst = list()
rLst = list()
for k, grp in enumerate(grpLst):
    varG = list(set(dfG.columns.tolist()).intersection(dictVar[grp]))
    npca = min(len(varG), 10)
    if npca > 0:
        x = dfG[varG].values
        x[np.isnan(x)] = -1
        pca = decomposition.PCA(n_components=npca)
        pca.fit(x)
        xx = pca.transform(x)
        r = pca.explained_variance_ratio_
        ind = np.where(r > 0.1)[0]
        for k in ind:
            pcaNameLst.append('{}_PCA{}'.format(grp, k))
            matPcaLst.append(xx[:, k])
            rLst.append(r[k])
matPca = np.stack(matPcaLst,axis=-1)


# plant tree
varLst = pcaNameLst
# x = dfG.astype(np.float32).values
x = matPca
y = mat[:, 0]
x[np.isnan(x)] = -1
y[np.isnan(y)] = -1
clf = sklearn.tree.DecisionTreeRegressor(
    max_leaf_nodes=20, min_samples_leaf=0.1)
clf = clf.fit(x, y)
tree = clf.tree_

# save a child tab
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
fileC = os.path.join(saveFolder, 'childrenTab.csv')
tabC = np.stack([tree.children_left, tree.children_right]).T
np.savetxt(fileC, tabC, fmt='%d', delimiter=',')

# plot and save figures


def plotLeaf(nodeId, indInput, title):
    fig, ax = plt.subplots(1, 1, figsize=[6, 3])
    lat = dfG['LAT_GAGE'][indInput].values
    lon = dfG['LNG_GAGE'][indInput].values
    data = y[indInput]
    axplot.mapPoint(ax, lat, lon, data, vRange=[0, 1], s=20)
    ax.set_title(title)
    # fig.show()
    figName = 'node{}.png'.format(nodeId)
    plt.savefig(os.path.join(saveFolder, figName))
    return fig


def plotNode(nodeId, indInput, indLeft, indRight, title):
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=[9, 3])
    # cdf
    ax = fig.add_subplot(gs[0, 0])
    cLst = 'gbr'
    y0 = y[indInput]
    y1 = y[indLeft]
    y2 = y[indRight]
    labLst = ['parent {:.2f}'.format(np.nanmean(y0)),
              'left {:.2f} (*)'.format(np.nanmean(y1)),
              'right {:.2f} (o)'.format(np.nanmean(y2))]
    dataLst = [y0, y1, y2]
    for kk, data in enumerate(dataLst):
        xSort = np.sort(data[~np.isnan(data)])
        yRank = np.arange(1, len(xSort)+1) / float(len(xSort))
        ax.plot(xSort, yRank, color=cLst[kk], label=labLst[kk])
        ax.set_xlim([0, 1])
    ax.legend(loc='best', frameon=False)
    # map
    ax = fig.add_subplot(gs[0, 1:])
    for indTemp, sty in zip([indLeft, indRight], '*o'):
        lat = dfG['LAT_GAGE'][indTemp].values
        lon = dfG['LNG_GAGE'][indTemp].values
        data = y[indTemp]
        axplot.mapPoint(ax, lat, lon, data, vRange=[0, 1], marker=sty, s=20)
    # fig.suptitle(title)
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    figName = 'node{}.png'.format(nodeId)
    plt.savefig(os.path.join(saveFolder, figName))
    return fig


matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
plt.tight_layout()

strLst, nodeLst, leaf, label = cart.TraverseTree(tree, x, Fields=varLst)
nn = len(nodeLst)
for k in range(nn):
    ind = np.array(nodeLst[k])
    th = tree.threshold[k]
    ix = tree.feature[k]
    if ix == -2:
        title = 'leaf#{}'.format(k)
        plotLeaf(k, ind,  title)
    else:
        indL = ind[np.where(x[ind, ix] <= th)[0]]
        indR = ind[np.where(x[ind, ix] > th)[0]]
        title = 'node#{} {} < {:.3f}'.format(k, varLst[ix], th)
        plotNode(k, ind, indL, indR, title)
