from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn, transform
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

codeLst = sorted(usgs.newC)
# dataName = 'nbWT'
dataName = 'nbW_norm'
wqData = waterQuality.DataModelWQ(dataName)
siteNoLst = wqData.info.siteNo.unique()

codeLst = usgs.newC
icLst = [wqData.varC.index(code) for code in codeLst]
data = wqData.c[:, np.array(icLst)]
mtdLst = wqData.extractVarMtd(codeLst)
dataNorm, stat = transform.transInAll(data, mtdLst)
info = wqData.info

code = '00660'
ic = codeLst.index(code)
fig, axes = plt.subplots(2, 1, figsize=(6, 8))
for siteNo in siteNoLst[1:2]:
    indS = info[info['siteNo'] == siteNo].index.values
    yr = utils.sortData(data[indS, ic])
    yn = utils.sortData(dataNorm[indS, ic])
    x = np.arange(len(yr))/len(yr)
    _ = axes[0].plot(x, yr, 'k-', alpha=0.2)
    _ = axes[1].plot(x, yn, 'k-', alpha=0.2)
shortName = usgs.codePdf.loc[code]['shortName']
axes[1].set_ylim([-0.2, 1.2])
axes[0].set_title('{} {} CDFs '.format(code, shortName))
axes[1].set_title('{} {} CDFs after normalization '.format(code, shortName))
fig.show()

siteNo = '01111500'
df = waterQuality.readSiteTS(siteNo, [code], freq='D').dropna()
dfC, dfCF = usgs.readSample(siteNo, codeLst=[code], flag=2)
dfC = dfC.resample('W-TUE').mean()
dfCF = dfCF.fillna(0)
dfCFW = dfCF.resample('W-TUE').mean()
dfCFW = dfCFW.fillna(0)
dfCFW[dfCFW != 0] = 1
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
t = dfC.index
v = dfC[code].values
flag = dfCFW[code+'_cd'].values
ax.plot(t[flag == 0], v[flag == 0], 'r*')
ax.plot(t[flag != 0], v[flag != 0], 'k*')
fig.show()

data = df.values
c1 = np.nanpercentile(data, 10)
c2 = np.nanpercentile(data, 90)
(data-c1)/(c2-c1)
