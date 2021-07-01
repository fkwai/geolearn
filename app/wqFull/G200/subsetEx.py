
from hydroDL.data import dbBasin, usgs
import json
import os
from hydroDL import kPath
import numpy as np
from hydroDL.post import axplot, figplot

sd = '1982-01-01'
ed = '2018-12-31'
dataName = 'G200'
dictSiteName = 'dict{}.json'.format(dataName)
dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, dictSiteName)) as f:
    dictSite = json.load(f)
siteNoLst = dictSite['rmTK']
# DF = dbBasin.DataFrameBasin.new(dataName, siteNoLst, sdStr=sd, edStr=ed)
DF = dbBasin.DataFrameBasin(dataName)

# print site count
for key in dictSite.keys():
    print(key, len(dictSite[key]))


seed = 0
rate = 0.2
rng = np.random.default_rng(seed)
mask = np.ones([len(DF.t), len(DF.siteNoLst)]).astype(bool)

for indS, siteNo in enumerate(DF.siteNoLst):
    obsB = np.any(~np.isnan(DF.c[:, indS, :]), axis=1)
    obsD = np.where(obsB)[0]
    nPick = int(sum(obsB*rate))
    # ind = rng.choice(obsD, nPick, replace=False)
    ind = obsD[-nPick:]
    mask[ind, indS] = False

for indS, siteNo in enumerate(DF.siteNoLst):
    indAll = np.arange(len(DF.t))
    nPick = int(len(DF.t)*rate)
    ind = rng.choice(indAll, nPick, replace=False)
    mask[ind, indS] = False


a = DF.extractSubset(DF.c, 'rmRT20')
b = DF.extractSubset(DF.c, 'pkRT20')
dataBox = list()
for code in DF.varC:
    siteNoCode = [x for x in dictSite[code] if x in DF.siteNoLst]
    indC = DF.varC.index(code)
    temp = list()
    for siteNo in siteNoCode:
        indS = DF.siteNoLst.index(siteNo)
        n1 = np.sum(~np.isnan(a[:, indS, indC]))
        n2 = np.sum(~np.isnan(b[:, indS, indC]))
        temp.append(n2/n1)
    dataBox.append(np.array(temp))
labLst1 = ['{}\n{}'.format(usgs.codePdf.loc[code]
                           ['shortName'], code) for code in DF.varC]
fig, ax = figplot.boxPlot(dataBox, cLst='br', label1=labLst1,
                          figsize=(6, 4), yRange=[0, 1])
fig.show()
