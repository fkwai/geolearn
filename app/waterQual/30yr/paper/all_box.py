import importlib
from hydroDL import kPath, utils
from hydroDL.app import waterQuality as wq
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import scipy
from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec

dirSel = os.path.join(kPath.dirData, 'USGS', 'inventory', 'siteSel')
with open(os.path.join(dirSel, 'dictRB_Y30N5.json')) as f:
    dictSite = json.load(f)

codeLst = sorted(usgs.newC)
ep = 500
reTest = False
siteNoLst = dictSite['comb']
nSite = len(siteNoLst)

# load all sequence
outNameLSTM = '{}-{}-{}-{}'.format('rbWN5', 'comb', 'QTFP_C', 'comb-B10')
dictLSTM, dictWRTDS, dictObs = wq.loadModel(
    siteNoLst, outNameLSTM, codeLst)
corrMat, rmseMat = wq.dictErr(dictLSTM, dictWRTDS, dictObs, codeLst)

# load basin attributes
dfG = gageII.readData(siteNoLst=siteNoLst)
dfG = gageII.updateRegion(dfG)
dfG = gageII.updateCode(dfG)


# significance test
dfS = pd.DataFrame(index=codeLst, columns=['rmse', 'corr'])
for k, code in enumerate(codeLst):
    a = corrMat[:, k, 1]
    b = corrMat[:, k, 2]
    aa, bb = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'corr'] = p
    a = rmseMat[:, k, 1]
    b = rmseMat[:, k, 2]
    aa, bb = utils.rmNan([a, b], returnInd=False)
    s, p = scipy.stats.ttest_ind(aa, bb)
    # s, p = scipy.stats.wilcoxon(aa, bb)
    dfS.at[code, 'rmse'] = p


# plot box corr
labLst1 = list()
codePlot = sorted(usgs.newC)
codePlot.remove('00950')
for code in codePlot:
    lab = '{}\n{}\n{:.0e}'.format(
        usgs.codePdf.loc[code]['shortName'], code, dfS.loc[code]['corr'])
    labLst1.append(lab)
labLst2 = ['LSTM', 'WRTDS']
dataBox = list()
for code in codePlot:
    temp = list()
    for i in [1, 2]:
        temp.append(corrMat[:, codeLst.index(code), i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5, cLst='rb',
                      label2=labLst2, figsize=(16, 5), yRange=[0, 1])
fig.show()
dirFig = r'C:\Users\geofk\work\paper\waterQuality'
fig.savefig(os.path.join(dirFig, 'box_all'))


a = np.log(np.log(1/dfS['corr'].values.astype(float)))
b = np.log(np.nanmedian(corrMat[:, :, 1], axis=0))

fig, ax = plt.subplots(1, 1)
for k in range(len(codeLst)):
    ax.text(b[k], a[k], usgs.codePdf.loc[codeLst[k]]['shortName'])
ax.plot(b, a, '*')
# ax.set_xlim([0.2, 1.2])
ax.set_ylim([-1.5, 3])
fig.show()

np.nanmean(corrMat[:, :, 1], axis=0)-np.nanmean(corrMat[:, :, 2], axis=0)
np.nanmedian(corrMat[:, :, 1], axis=0)-np.nanmedian(corrMat[:, :, 2], axis=0)
