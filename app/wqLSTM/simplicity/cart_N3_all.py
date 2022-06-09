

import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
from hydroDL.app import waterQuality, cart
import matplotlib.gridspec as gridspec
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
lat, lon = DF.getGeo()
# remove some attrs
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
dfG = gageII.updateCode(dfG)

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

# for a code
for code in codeLst:
    saveFolder = r'C:\Users\geofk\work\waterQuality\paper\G200\simplicity\{}'.format(
        code)
    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)
    ic = codeLst.index(code)
    [mat], indS = utils.rmNan([matQS[:, ic]])
    siteNoCode = [DF.siteNoLst[ind] for ind in indS]
    dfGC = dfG.loc[siteNoCode][colLst]

    # plant tree
    varLst = dfGC.columns.tolist()
    # x = dfG.astype(np.float32).values
    x = dfGC.values
    y = mat
    x[np.isnan(x)] = -99
    # y[np.isnan(y)] = -99
    clf = sklearn.tree.DecisionTreeRegressor(
        max_leaf_nodes=20, min_samples_leaf=0.1, max_depth=3)
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
        fig = plt.figure(figsize=[6, 3])
        gs = gridspec.GridSpec(1, 1)
        lat = dfGC['LAT_GAGE'][indInput].values
        lon = dfGC['LNG_GAGE'][indInput].values
        data = y[indInput]
        ax = mapplot.mapPoint(fig, gs[0:1, 0:1], lat,
                              lon, data, vRange=[0, 1], s=20)
        ax.set_title(title)
        # fig.show()
        figName = 'node{}.png'.format(nodeId)
        plt.savefig(os.path.join(saveFolder, figName))
        return fig

    def plotNode(nodeId, indInput, indLeft, indRight, title):
        gs = gridspec.GridSpec(1, 5)
        fig = plt.figure(figsize=[9, 3])
        # cdf
        ax = fig.add_subplot(gs[0, 2])
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
        for indTemp, sty, gsT in zip([indLeft, indRight], '*o', [gs[0, 0:2], gs[0, 3:]]):
            lat = dfGC['LAT_GAGE'][indTemp].values
            lon = dfGC['LNG_GAGE'][indTemp].values
            data = y[indTemp]
            mapplot.mapPoint(fig, gsT, lat, lon, data,
                             vRange=[0, 1], marker=sty, s=20)
        # fig.suptitle(title)
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        figName = 'node{}.png'.format(nodeId)
        plt.savefig(os.path.join(saveFolder, figName))
        return fig

    strLst, nodeLst, leaf, label = cart.TraverseTree(tree, x, Fields=varLst)
    nn = len(nodeLst)
    for k in range(nn):
        ind = np.array(nodeLst[k])
        th = tree.threshold[k]
        ix = tree.feature[k]
        print(k, ix)
        if ix == -2:
            title = 'leaf#{}'.format(k)
            plotLeaf(k, ind,  title)
        else:
            indL = ind[np.where(x[ind, ix] <= th)[0]]
            indR = ind[np.where(x[ind, ix] > th)[0]]
            title = 'node#{} {} < {:.3f}'.format(k, varLst[ix], th)
            plotNode(k, ind, indL, indR, title)
