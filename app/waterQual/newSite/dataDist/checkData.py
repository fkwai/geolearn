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
dataName = 'nbW'
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
for siteNo in siteNoLst:
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
