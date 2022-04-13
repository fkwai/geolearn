
from hydroDL import kPath
from sklearn.decomposition import PCA
import sklearn
import torch.nn.functional as F
import torch.nn as nn
import random
import os
from hydroDL.model import trainBasin, crit, waterNetTestC, waterNetTest
from hydroDL.data import dbBasin, gageII, usgs
import numpy as np
import torch
import pandas as pd
import importlib
from hydroDL.utils import torchUtils
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from hydroDL.model.waterNet import WaterNet0119, sepPar, convTS
from hydroDL import utils
importlib.reload(waterNetTestC)
# extract data
codeLst = ['00600', '00660', '00915', '00925', '00930', '00935', '00945']

siteNo = '09163500'

df = dbBasin.io.readSiteTS(siteNo, codeLst,)
fig, axes = figplot.multiTS(df.index, df.values)
fig.show()

dfC, dfF = usgs.readSample(siteNo, codeLst=codeLst, flag=2, csv=True)
fig, axes = figplot.multiTS(dfC.index, dfC.values)
fig.show()

dfC = usgs.removeFlag(dfC, dfF)
fig, axes = figplot.multiTS(dfC.index, dfC.values)
fig.show()

fileC = os.path.join(kPath.dirData, 'USGS', 'sample', siteNo)
dfC = usgs.read.readUsgsText(fileC, dataType='sample')
dfC = dfC.set_index('date')
if codeLst is None:
    codeSel = [x for x in dfC.columns.tolist() if x.isdigit()]
else:
    codeSel = list(set(codeLst) & set(dfC.columns.tolist()))
codeSel_cd = [code + '_cd' for code in codeSel]
dfC = dfC[codeSel+codeSel_cd].dropna(how='all')
dfC1 = dfC[codeSel]
dfC2 = dfC[codeSel_cd]
bx = dfC1.notna().values & dfC2.isna().values
dfC2.values[bx] = 'x'
dfC2 = dfC2.fillna('')
bDup = dfC.index.duplicated(keep=False)
indUni = dfC.index[~bDup]
indDup = dfC.index[bDup].unique()
indAll = dfC.index.unique()
dfO1 = pd.DataFrame(index=indAll, columns=codeSel)
dfO2 = pd.DataFrame(index=indAll, columns=codeSel_cd)
dfO1.loc[indUni] = dfC1.loc[indUni][codeSel]
dfO2.loc[indUni] = dfC2.loc[indUni][codeSel_cd]
for ind in indDup:
    temp1 = dfC1.loc[ind]
    temp2 = dfC2.loc[ind]
    for code in codeSel:
        if 'x' in temp2[code+'_cd'].tolist():
            dfO1.loc[ind][code] = temp1[code][temp2[code+'_cd']
                                                == 'x'].mean()
            if temp2[code+'_cd'].tolist().count('x') > 1:
                dfO2.loc[ind][code+'_cd'] = 'X'
            else:
                dfO2.loc[ind][code+'_cd'] = 'x'
        else:
            dfO1.loc[ind][code] = temp1[code].mean()
            dfO2.loc[ind][code+'_cd'] = ''.join(temp2[code+'_cd'])

    dfO3 = pd.DataFrame(
        index=dfO2.index, columns=dfO2.columns, dtype=int)
    dfO3[(dfO2 == 'x') | (dfO2 == 'X')] = 0
    dfO3[(dfO2 != 'x') & (dfO2 != 'X') & (dfO2.notna())] = 1
    dfO2 = dfO3
codeLst_cd = [code + '_cd' for code in codeLst]