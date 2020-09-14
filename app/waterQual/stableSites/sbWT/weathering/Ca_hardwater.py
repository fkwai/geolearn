from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

code = '00915'
trainSet = '{}-Y1'.format(code)
testSet = '{}-Y2'.format(code)
outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnq', trainSet)
# outName = '{}-{}-{}-{}'.format(dataName, code, 'plain', trainSet)
siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
corrMat = np.full([len(siteNoLst),  2], np.nan)
rmseMat = np.full([len(siteNoLst),  2], np.nan)
ic = wqData.varC.index(code)
yP, ycP = basins.testModel(
    outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
ind = wqData.subset[testSet]
bdate = wqData.info.iloc[ind]['date'].values > np.datetime64('1980-01-01')
o = wqData.c[-1, ind[bdate], ic]
p = yP[-1, bdate, 1]

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.plot(o, p, '*')
fig.show()

hardMat = np.ndarray([4, 4])
theLst = [0, 60, 120, 180, 500]
bMat = np.full([len(o), 2], np.nan)
for i in range(4):
    v1 = theLst[i]
    v2 = theLst[i+1]
    ind1 = np.where((o > v1) & (o <= v2))[0]
    ind2 = np.where((p > v1) & (p <= v2))[0]
    bMat[ind1, 0] = i
    bMat[ind2, 1] = i
for i in range(4):
    n = len(np.where(bMat[:, 0] == i)[0])
    print(n, n/len(o)*100)
    for j in range(4):
        nn = len(np.where((bMat[:, 0] == i) & (bMat[:, 1] == j))[0])
        hardMat[j, i] = nn/n

fig, ax = plt.subplots()
axplot.plotHeatMap(
    ax, hardMat*100, labLst=['soft', 'moderate', 'hard', 'extreme'])
fig.tight_layout()
fig.show()
