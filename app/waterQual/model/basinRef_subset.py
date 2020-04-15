from hydroDL.master import basins
from hydroDL.app import waterQuality, relaCQ
from hydroDL import kPath, utils
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('basinRef')

# rmnan in q
q = wqData.q[:, :, 0]
info = wqData.info
len(np.where(np.isnan(q).all(axis=0))[0])
len(np.where(np.isnan(q).any(axis=0))[0])
len(wqData.info)  # nan in Q - 3% all nan, 7% any nan
indR = np.where(np.isnan(q).any(axis=0))[0]
indK = np.where(~np.isnan(q).any(axis=0))[0]
# purify q
infoK = info.iloc[indK]
dfSite = waterQuality.countSite(infoK)
ind1 = dfSite[dfSite['pRank'] <= 0.5].index.values
ind2 = dfSite[dfSite['pRank'] > 0.5].index.values
wqData.saveSubset(['pQ-F50', 'pQ-L50'], [ind1, ind2])
# yr after purify q
indYr = waterQuality.indYr(infoK)
yrLst = ['Y80', 'Y90', 'Y00', 'Y10']
wqData.saveSubset(['pQ-'+x for x in yrLst], indYr)
indYrCmp = list()
indAll = infoK.index.values
for ind in indYr:
    indYrCmp.append(np.setdiff1d(indAll, ind))
wqData.saveSubset(['pQ-rm'+x for x in yrLst], indYrCmp)
# validate
# d=wqData.info.iloc[wqData.subset['pQ-Y00']]['date']
# np.sort(pd.DatetimeIndex(d).year.unique())

# regional / area subsets
varG = ['DRAIN_SQKM', 'ECO2_BAS_DOM', 'NUTR_BAS_DOM',
        'HLR_BAS_DOM_100M']
tabG = gageII.readData(varLst=varG, siteNoLst=wqData.siteNoLst)
info = wqData.info
indF50 = wqData.indByRatio(0.5)
indL50 = wqData.indByRatio(0.5, first=False)
wqData.saveSubset(['first50', 'last50'], [indF50, indL50])

siteNoLst = list()
nameLst = list()
siteNoLst.append(tabG[tabG['DRAIN_SQKM'] < 10].index.tolist())
nameLst.append('areaLT10')
siteNoLst.append(tabG[tabG['DRAIN_SQKM'] > 2500].index.tolist())
nameLst.append('areaGT2500')
siteNoLst.append(tabG[tabG['ECO2_BAS_DOM'] == 5.3].index.tolist())
nameLst.append('eco0503')
siteNoLst.append(tabG[tabG['ECO2_BAS_DOM'] == 9.2].index.tolist())
nameLst.append('eco0902')
siteNoLst.append(tabG[tabG['NUTR_BAS_DOM'] == 6].index.tolist())
nameLst.append('nutr06')
siteNoLst.append(tabG[tabG['NUTR_BAS_DOM'] == 8].index.tolist())
nameLst.append('nutr08')

indLst = list()
subsetLst = list()
for siteNo, name in zip(siteNoLst, nameLst):
    indSite = info[info['siteNo'].isin(siteNo)].index.values
    indLst.append(np.intersect1d(indF50, indSite))
    subsetLst.append(name+'-F50')
    indLst.append(np.intersect1d(indL50, indSite))
    subsetLst.append(name+'-L50')

wqData.saveSubset(subsetLst, indLst)
