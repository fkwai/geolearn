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

# a cdf for rsq of seasonality and linearity
codeLst2 = ['00915', '00925', '00930', '00935', '00940', '00945',
            '00955', '70303', '80154']
[nfy, nfx] = [4, 2]
fig, axes = plt.subplots(4, 2)
for k, code in enumerate(codeLst2):
    j, i = utils.index2d(k, 4, 2)
    indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
    ic = codeLst.index(code)
    axplot.plotCDF(axes[j, i], [corrMat[indS, ic, 1]**2, corrMat[indS, ic, 2]**2],
                   legLst=['LSTM', 'WRTDS'])
    axes[j, i].set_title(code)
fig.show()

code = '00405'
indS = [siteNoLst.index(siteNo) for siteNo in dictSite[code]]
ic = codeLst.index(code)
fig, ax = plt.subplots(1, 1)
ax.plot(corrMat[indS, ic, 1]**2, corrMat[indS, ic, 2]**2, '*')
fig.show()

np.sum(corrMat[indS, ic, 1]**2 > corrMat[indS, ic, 2]**2)


np.sum(corrMat[indS, ic, 1]**2 > corrMat[indS, ic, 2]**2)
np.sum(~np.isnan(corrMat[indS, ic, 2]))
np.nanmedian(corrMat[indS, ic, 2])

temp1 = corrMat[indS, ic, 1]
temp2 = corrMat[indS, ic, 2]
ind1 = np.where(corrMat[indS, ic, 1]**2 > 0.5)[0]
ind2 = np.where(corrMat[indS, ic, 1]**2 <= 0.5)[0]

np.nanmedian(temp1)
np.nanmedian(temp2)
np.nanmean(temp1)
np.nanmean(temp2)

np.nanmedian(temp1[ind1])
np.nanmedian(temp2[ind1])
np.nanmedian(temp1[ind2])
np.nanmedian(temp2[ind2])

np.nanmean(temp1[ind1])
np.nanmean(temp2[ind1])
np.nanmean(temp1[ind2])
np.nanmean(temp2[ind2])

len(np.where(temp1>temp2)[0])
