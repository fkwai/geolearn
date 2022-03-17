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
    dictLSTMLst = list()
    # LSTM
    labelLst = ['QTFP_C']
    for label in labelLst:
        dictLSTM = dict()
        trainSet = 'comb-B10'
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        for k, siteNo in enumerate(siteNoLst):
            print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
            df = basins.loadSeq(outName, siteNo)
            dictLSTM[siteNo] = df
        dictLSTMLst.append(dictLSTM)
    # WRTDS
    dictWRTDS = dict()
    dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat',
                            'WRTDS-W', 'B10', 'output')
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        saveFile = os.path.join(dirWRTDS, siteNo)
        df = pd.read_csv(saveFile, index_col=None).set_index('date')
        # df = utils.time.datePdf(df)
        dictWRTDS[siteNo] = df
    # Observation
    dictObs = dict()
    for k, siteNo in enumerate(siteNoLst):
        print('\t site {}/{}'.format(k, len(siteNoLst)), end='\r')
        df = waterQuality.readSiteTS(
            siteNo, varLst=['00060']+codeLst, freq='W')
        dictObs[siteNo] = df

    # calculate correlation
    tt = np.datetime64('2010-01-01')
    t0 = np.datetime64('1980-01-01')
    ind1 = np.where((df.index.values < tt) & (df.index.values >= t0))[0]
    ind2 = np.where(df.index.values >= tt)[0]
    dictLSTM = dictLSTMLst[0]
    corrMat = np.full([len(siteNoLst), len(codeLst), 3], np.nan)
    for ic, code in enumerate(codeLst):
        for siteNo in dictSite[code]:
            indS = siteNoLst.index(siteNo)
            v1 = dictLSTM[siteNo][code].iloc[ind2].values
            v2 = dictWRTDS[siteNo][code].iloc[ind2].values
            v3 = dictObs[siteNo][code].iloc[ind2].values
            vv1, vv2, vv3 = utils.rmNan([v1, v2, v3], returnInd=False)
            rmse1, corr1 = utils.stat.calErr(vv1, vv2)
            rmse2, corr2 = utils.stat.calErr(vv1, vv3)
            rmse3, corr3 = utils.stat.calErr(vv2, vv3)
            corrMat[indS, ic, 0] = corr1
            corrMat[indS, ic, 1] = corr2
            corrMat[indS, ic, 2] = corr3

    # load basin attributes
    regionLst = ['ECO2_BAS_DOM', 'NUTR_BAS_DOM',
                 'HLR_BAS_DOM_100M', 'PNV_BAS_DOM']
    dfG = gageII.readData(siteNoLst=siteNoLst)
    fileT = os.path.join(gageII.dirTab, 'lookupPNV.csv')
    tabT = pd.read_csv(fileT).set_index('PNV_CODE')
    for code in range(1, 63):
        siteNoTemp = dfG[dfG['PNV_BAS_DOM'] == code].index
        dfG.at[siteNoTemp, 'PNV_BAS_DOM2'] = tabT.loc[code]['PNV_CLASS_CODE']
    dfG = gageII.updateCode(dfG)

code = '00915'
ic1 = codeLst.index('00915')
ic2 = codeLst.index('00955')
matAll = corrMat[:, ic1, 1]**2-corrMat[:, ic2, 1]**2
[mat], indS = utils.rmNan([matAll])
siteNoCode = [siteNoLst[ind] for ind in indS]
dfGC = dfG.loc[siteNoCode]


def subTree(indInput, varLst):
    x = dfGC.iloc[indInput][varLst].values.astype(float)
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
    lat = dfGC['LAT_GAGE'][indInput]
    lon = dfGC['LNG_GAGE'][indInput]
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
colLst = dfGC.columns.tolist()
ind0 = np.arange(len(siteNoCode))
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
