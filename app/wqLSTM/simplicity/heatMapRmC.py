import matplotlib
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
import sklearn.tree
import matplotlib.gridspec as gridspec


# investigate correlation between simlicity and basin attributes.
# remove carbon - less obs, high corr

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

#  gageII
dfG = gageII.readData(siteNoLst=DF.siteNoLst)
dfG = gageII.updateCode(dfG)
lat = dfG['LAT_GAGE'].values
lon = dfG['LNG_GAGE'].values
# remove some attrs
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

# remove carbon
iC = codeLst.index('00681')
codeLst.remove('00681')
matQ = np.delete(matQ, iC, 1)
matS = np.delete(matS, iC, 1)
matQS = np.delete(matQS, iC, 1)
labX = list()
for code in codeLst:
    temp = usgs.codePdf.loc[code]['shortName']
    labX.append('{} {}'.format(temp, code))

data = dfG[colLst].values
corrQS = np.full([len(colLst), len(codeLst)], np.nan)
for j in range(len(colLst)):
    for i in range(len(codeLst)):
        corrQS[j, i] = utils.stat.calCorr(data[:, j], matQS[:, i])

indF = np.unique(np.where(np.abs(corrQS) > 0.4)[0])
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
fig, ax = plt.subplots(1, 1,figsize=(18,10))
labLst = [labX, [colLst[ind] for ind in indF]]
axplot.plotHeatMap(ax, corrQS[indF, :].T*100, labLst=labLst)
plt.tight_layout()
fig.show()
dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
plt.savefig(os.path.join(dirPaper, 'heatMapG2Simp'))
plt.savefig(os.path.join(dirPaper, 'heatMapG2Simp.svg'))

gageII.getVariableDict( [colLst[ind] for ind in indF])


# good corr appears in C
iC = codeLst.index('00955')
[v], indV = utils.rmNan([corrQS[:, iC]])
indSel = indV[np.argsort(-abs(v))[:10]]
dictTop = dict()
for x in indSel:
    dictTop[colLst[x]] = corrQS[x, iC]

ind = indSel[0]
aa = matQS[:, iC]
bb = data[:, ind]
[a, b], indS = utils.rmNan([aa, bb])
fig = plt.figure(figsize=[12, 6])
gs = gridspec.GridSpec(2, 2)
ax = mapplot.mapPoint(fig, gs[0, 0], lat[indS], lon[indS], a, s=20)
code = codeLst[iC]
usgs.codePdf.loc[code]['shortName']
ax.set_title('simplicity of {}'.format(labX[iC]))
ax = mapplot.mapPoint(fig, gs[1, 0], lat[indS], lon[indS], b, s=20)
ax.set_title('{}'.format(colLst[ind]))
ax = fig.add_subplot(gs[:, 1])
ax.plot(b, a, '*')
ax.set_xlabel(colLst[ind])
ax.set_ylabel(labX[iC])
fig.show()

for item in dictTop:
    print('{:.2f} {}'.format(dictTop[item], item))
