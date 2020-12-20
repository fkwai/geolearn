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

# load gageII
dfGeo = gageII.readData()
dfGeo = gageII.updateCode(dfGeo)
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
mat = dfCorr.loc[siteNoLst][codeLst].values[:, 0]
dfG = dfGeo.loc[siteNoLst]

# remove columns
rmColLst = ['REACHCODE', 'STANAME']
for yr in range(1950, 2010):
    rmColLst.append('PPT{}_AVG'.format(yr))
    rmColLst.append('TMP{}_AVG'.format(yr))
for yr in range(1900, 2010):
    rmColLst.append('wy{}'.format(yr))
monthLst = ['JAN', 'FEB', 'APR', 'MAY', 'JUN',
            'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
for m in monthLst:
    rmColLst.append('{}_PPT7100_CM'.format(m))
    rmColLst.append('{}_TMP7100_DEGC'.format(m))
dfG = dfG.drop(rmColLst, axis=1)

# plant tree
varLst = dfG.columns.tolist()
# x = dfG.astype(np.float32).values
x = dfG.values
y = mat[:, 0]
x[np.isnan(x)] = -99
y[np.isnan(y)] = -99
clf = sklearn.tree.DecisionTreeRegressor(
    max_leaf_nodes=20, min_samples_leaf=20)
clf = clf.fit(x, y)
tree = clf.tree_

# save a child tab
saveFolder = os.path.join(dirTree, 'test2')
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)
fileC = os.path.join(saveFolder, 'childrenTab.csv')
tabC = np.stack([tree.children_left, tree.children_right]).T
np.savetxt(fileC, tabC, fmt='%d', delimiter=',')

# plot and save figures


def TraverseTree(regTree, Xin, Fields=None):
    nnode = regTree.node_count
    if Fields is not None:
        featurename = [Fields[i] for i in regTree.feature]
    else:
        featurename = ["Fields%i" % i for i in regTree.feature]
    string = ['']*nnode
    nodeind = [None]*nnode
    nind = Xin.shape[0]
    leaf = []
    label = np.zeros([nind])

    def recurse(tempstr, node, Xtemp, indtemp):
        string[node] = "node#%i: " % node+tempstr
        nodeind[node] = indtemp
        if (regTree.threshold[node] != -2):
            if regTree.children_left[node] != -1:
                tempstr = tempstr + \
                    " ( " + featurename[node] + " <= " + \
                    "%.3f" % (regTree.threshold[node]) + " ) ->\n "
                indlocal = np.where(
                    Xtemp[:, regTree.feature[node]] <= regTree.threshold[node])[0]
                indleft = [indtemp[i] for i in indlocal]
                Xleft = Xtemp[indlocal, :]
                recurse(tempstr, regTree.children_left[node], Xleft, indleft)
            if regTree.children_right[node] != -1:
                tempstr = " ( " + featurename[node] + " > " + \
                    "%.3f" % (regTree.threshold[node]) + " ) ->\n "
                indlocal = np.where(
                    Xtemp[:, regTree.feature[node]] > regTree.threshold[node])[0]
                indright = [indtemp[i] for i in indlocal]
                Xright = Xtemp[indlocal, :]
                recurse(
                    tempstr, regTree.children_right[node], Xright, indright)
        else:
            leaf.append(node)
            label[indtemp] = node
    recurse('', 0, Xin, range(0, nind))
    return string, nodeind, leaf, label


def plotLeaf(nodeId, indInput, title):
    fig, ax = plt.subplots(1, 1, figsize=[6, 3])
    lat = dfG['LAT_GAGE'][indInput].values
    lon = dfG['LNG_GAGE'][indInput].values
    data = mat[indInput]
    axplot.mapPoint(ax, lat, lon, data, vRange=[0, 1], s=20)
    ax.set_title(title)
    # fig.show()
    figName = 'node{}.png'.format(nodeId)
    plt.savefig(os.path.join(saveFolder,figName))


def plotNode(nodeId, indInput, indLeft, indRight, title):
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize=[9, 3])
    # cdf
    ax = fig.add_subplot(gs[0, 0])
    cLst = 'gbr'
    labLst = ['parent', 'left (*)', 'right (o)']
    y0 = mat[indInput]
    y1 = mat[indLeft]
    y2 = mat[indRight]
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
        data = mat[indTemp]
        axplot.mapPoint(ax, lat, lon, data, vRange=[0, 1], marker=sty, s=20)
    # fig.suptitle(title)
    ax.set_title(title)
    plt.tight_layout()
    figName = 'node{}.png'.format(nodeId)
    plt.savefig(os.path.join(saveFolder,figName))


strLst, nodeLst, leaf, label = TraverseTree(tree, x, Fields=varLst)
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

indInput = ind
indLeft = indL
indRight = indR
