import importlib
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import itertools

# MAKE USE OF OLD RESULT
fileName = r'C:\Users\geofk\work\database\USGS\inventory\codeCombCount'
dfCount = pd.read_csv(fileName, header=None, index_col=0)

dictCount = dict()

for k in range(len(dfCount)):
    print(k, len(dfCount))
    name = dfCount.index[k]
    count = dfCount.loc[name][1]
    codeLst = name.split('-')
    codeLst.sort()
    nameNew = '-'.join(codeLst)
    dictCount[nameNew] = count

dictOut = dict()
dictN = dict()
dictB = dict()

for k, key in enumerate(dictCount):
    print(k, len(dictCount))
    codeLst = key.split('-')
    n = len(codeLst)
    for k in range(n):
        for subset in itertools.combinations(codeLst, k):
            keyOut = '-'.join(subset)
            if keyOut not in dictOut:
                dictOut[keyOut] = dictCount[key]
                dictN[keyOut] = len(subset)
                # add bc later
            else:
                dictOut[keyOut] = dictOut[keyOut]+dictCount[key]

tab = pd.DataFrame.from_dict(dictOut, orient='index')
tab = tab.sort_values(0, ascending=False)

tab = tab.rename(columns={0: 'count'})
nc = np.zeros(len(tab))
bc = np.zeros(len(tab))
for k, x in enumerate(tab.index):
    print(k)
    codeLst = x.split('-')
    nc[k] = len(x.split('-'))
    if len(set(codeLst)-set(usgs.chemLst)) == 0:
        bc[k] = 1
tab['nc'] = nc.astype(int)
tab['bc'] = bc.astype(int)


dirComb = r'C:\Users\geofk\work\database\USGS\inventory\codeComb'
tab.to_csv(os.path.join(dirComb, 'codeCombAll'), header=False)


x = '00915-00925-00930-00935-00940'
codeLst = x.split('-')
len(set(codeLst)-set(usgs.chemLst)) > 0

df = tab[(tab['bc'] == 1) & (tab['nc'] == 1)].head(20)
for n in range(2, 21):
    temp = tab[(tab['bc'] == 1) & (tab['nc'] == n)].head(20)
    df = df.append(temp)

dirComb = r'C:\Users\geofk\work\database\USGS\inventory\codeComb'
df.to_csv(os.path.join(dirComb, 'codeCombTop'), header=False)


[usgs.codePdf.loc[code]['shortName'] for code in usgs.chemLst]
usgs.codePdf.loc['00605']['shortName']
