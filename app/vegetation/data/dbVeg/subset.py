import hydroDL.data.dbVeg
from hydroDL.data import dbVeg
import importlib
import numpy as np
import json
import os

importlib.reload(hydroDL.data.dbVeg)
df = dbVeg.DataFrameBasin('single')

subsetName='5fold_0_train'
# subsetName='all'
df.loadSubset(subsetName)
dm=dbVeg.DataModelVeg(df, subsetName=subsetName)

# random subset 5 kfold
ind = np.array(range(len(df.siteIdLst)))
np.random.seed(0)
np.random.shuffle(ind)
n = 5
indLst = np.array_split(ind, n)
dictSubset = dict()
dictSubset['all'] = ind.tolist()
for k in range(n):
    indTest = sorted(indLst[k].tolist())
    indTrain = sorted(np.concatenate(indLst[:k] + indLst[k + 1 :]).tolist())
    dictSubset['{}fold_{}_train'.format(n, k)] = indTrain
    dictSubset['{}fold_{}_test'.format(n, k)] = indTest
folder = dbVeg.caseFolder('single')

with open(os.path.join(folder, 'subset.json'), 'w') as fp:
    json.dump(dictSubset, fp, indent=4)
