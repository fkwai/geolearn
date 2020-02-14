from hydroDL.data import usgs, gageII, transform
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.post import axplot
import pandas as pd
import numpy as np
import time
import os
import pickle
from scipy.stats import linregress
import importlib

dirUSGS = os.path.join(kPath.dirData, 'USGS')
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
dirCQ = os.path.join(kPath.dirWQ, 'C-Q')
fileSiteNoLst = os.path.join(dirInv, 'siteNoLst')
siteNoLst = pd.read_csv(fileSiteNoLst, header=None, dtype=str)[0].tolist()
doLog = False

t0 = time.time()
fileName = os.path.join(dirCQ, 'CQall')
if not os.path.exists(fileName):
    dictData = dict()
    errLst = list()
    for i, siteNo in enumerate(siteNoLst):
        csvC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv', siteNo)
        csvQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', 'csv', siteNo)
        if os.path.exists(csvC):
            dfC = pd.read_csv(csvC)
        else:
            dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst)
        if os.path.exists(csvQ):
            dfQ = pd.read_csv(csvQ)
        else:
            dfQ = usgs.readStreamflow(siteNo)
        if len(dfC.index) == 0:
            errLst.append(siteNo)
        pdf = pd.concat([dfC.set_index('date').dropna(
            how='all'), dfQ.set_index('date')], axis=1, join='inner')
        dictData[siteNo] = pdf
        print('\t {}/{} {:.2f}'.format(
            i, len(siteNoLst), time.time()-t0), end='\r')
    fileName = os.path.join(kPath.dirWQ, 'tempData', 'CQall')
    pickle.dump(dictData, open(fileName, 'wb'))
else:
    dictData = pickle.load(open(fileName, 'rb'))
print('read all C-Q data {:.2f}'.format(time.time()-t0))

# calculate slope
codeLst = waterQuality.codeLst
nSite = len(siteNoLst)
codeQ = '00060_00003'
slopeMat = np.full([nSite, len(codeLst)], np.nan)
nSampleMat = np.full([nSite, len(codeLst)], np.nan)
t0 = time.time()
for i, codeC in enumerate(codeLst):
    for j, siteNo in enumerate(siteNoLst):
        pdf = dictData[siteNo][[codeC, codeQ]].dropna()
        if len(pdf.index) > 1:
            if doLog is True:
                x = np.log10(pdf[codeQ].values)
                y = np.log10(pdf[codeC].values)
            else:
                x = pdf[codeQ].values
                y = pdf[codeC].values
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            slopeMat[j, i] = slope
            nSampleMat[j, i] = len(x)
        print('\t {} {} {}/{} {:.2f}'.format(i, codeC,
                                             j, nSite, time.time()-t0), end='\r')
# better save as csv
df = pd.DataFrame(data=slopeMat, index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
if doLog is True:
    df.to_csv(os.path.join(dirCQ, 'slope'))
else:
    df.to_csv(os.path.join(dirCQ, 'slopeLog'))
df = pd.DataFrame(data=nSampleMat, index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'nSample'))


# # 121 plot
# fig, ax = plt.subplots(1, 1)
# temp = slopeMat[~np.isnan(slopeMat)]
# ax.hist(temp, bins=200, range=[
#         np.percentile(temp, 5), np.percentile(temp, 95)])
# fig.show()


# plot time series
# normCLst = list()
# for k in range(len(dfC.columns)):
#     normC, stat = transform.transIn(dfC.values[:, k], mtd='norm')
#     if not np.isnan(normC).all():
#         normCLst.append(normC)
# normQ, stat = transform.transIn(dfQ.values, mtd='norm')
# fig, ax = plt.subplots(1, 1)
# axplot.plotTS(ax, dfQ.index.values, normQ, cLst=['gray'])
# axplot.plotTS(ax, dfC.index.values, normCLst, cLst='rbkgcmy')
# fig.show()
