from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
import scipy

# all gages
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
codeLst = sorted(usgs.codeLst)

doLst = list()
# doLst.append('calCount')
# doLst.append('calCountCorr')

if 'calCount' in doLst:
    # calculate number of samples (all, B2000, A2000)
    df0 = pd.DataFrame(index=siteNoLstAll, columns=codeLst)
    df1 = pd.DataFrame(index=siteNoLstAll, columns=codeLst)
    df2 = pd.DataFrame(index=siteNoLstAll, columns=codeLst)
    tBar = np.datetime64('2000-01-01')
    for k, siteNo in enumerate(siteNoLstAll):
        print(k)
        dfC = waterQuality.readSiteY(siteNo, codeLst)
        df0.loc[siteNo] = dfC.count()
        df1.loc[siteNo] = dfC[dfC.index < tBar].count()
        df2.loc[siteNo] = dfC[dfC.index >= tBar].count()
    df0.to_csv(os.path.join(dirInv, 'codeCount.csv'))
    df1.to_csv(os.path.join(dirInv, 'codeCount_B2000.csv'))
    df2.to_csv(os.path.join(dirInv, 'codeCount_A2000.csv'))

if 'calCount' in doLst:
    # find out two variables (hopefully one rock one bio) that are most related
    df0 = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                    dtype={'siteNo': str}, index_col='siteNo')
    df1 = pd.read_csv(os.path.join(dirInv, 'codeCount_B2000.csv'),
                    dtype={'siteNo': str}, index_col='siteNo')
    df2 = pd.read_csv(os.path.join(dirInv, 'codeCount_A2000.csv'),
                    dtype={'siteNo': str}, index_col='siteNo')
    nc = len(codeLst)
    dfLst = [df0, df1, df2]
    titleLst = ['all', 'B2000', 'C2000']
    for df, title in zip(dfLst, titleLst):
        matCorr = np.full([nc, nc], np.nan)
        for j, c1 in enumerate(codeLst):
            for i, c2 in enumerate(codeLst):
                v1 = df[c1].values
                v2 = df[c2].values
                # ind = np.where((v1 != 0) & (v2 != 0))[0]
                # corr, p = scipy.stats.spearmanr(v1[ind], v2[ind])
                corr, p = scipy.stats.spearmanr(v1, v2)
                # corr, p = scipy.stats.pearsonr(v1, v2)
                matCorr[j, i] = corr
        varNameLst = ['{} {}'.format(
            usgs.codePdf.loc[code]['shortName'], code) for code in codeLst]
        fig, ax = plt.subplots()
        axplot.plotHeatMap(ax, matCorr*100, varNameLst)
        ax.set_title('spearman correlation of {}'.format(title))
        fig.tight_layout()
        fig.show()
