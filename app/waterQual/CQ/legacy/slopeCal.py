from scipy.optimize import curve_fit
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

t0 = time.time()
fileName = os.path.join(dirCQ, 'CQall')
if not os.path.exists(fileName):
    dictData = dict()
    errLst = list()
    for i, siteNo in enumerate(siteNoLst):
        csvC = os.path.join(kPath.dirData, 'USGS', 'sample', 'csv', siteNo)
        csvQ = os.path.join(kPath.dirData, 'USGS', 'streamflow', 'csv', siteNo)
        dfC = usgs.readSample(siteNo, codeLst=waterQuality.codeLst)
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
pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
unitConv = 0.3048**3*365*24*60*60/1000**2
codeLst = waterQuality.codeLst
# codeLst = ['00955', '00940', '00915']

nSite = len(siteNoLst)
codeQ = '00060_00003'
pMat = np.full([nSite, len(codeLst), 4], np.nan)
nMat = np.full([nSite, len(codeLst)], np.nan)
t0 = time.time()
for i, codeC in enumerate(codeLst):
    for j, siteNo in enumerate(siteNoLst):
        pdf = dictData[siteNo][[codeC, codeQ]].dropna()
        if len(pdf.index) > 10:
            area = pdfArea.loc[siteNo].values[0]
            q = pdf[codeQ].values/area*unitConv
            c = pdf[codeC].values
            # slope model
            try:
                x1 = np.log(q)
                y1 = np.log(c)
                ind = np.where((~np.isnan(x1+y1)) & (~np.isinf(x1+y1)))
                a, b, r, p, std = linregress(x1[ind], y1[ind])
                sa = np.exp(b)
                sb = a
                pMat[j, i, 0:2] = [sa, sb]
            except:
                pass
            # kate model
            try:
                x2 = q
                y2 = 1/c
                ind = np.where((~np.isnan(x2+y2)) & (~np.isinf(x2+y2)))
                a, b, r, p, std = linregress(x2[ind], y2[ind])
                ceq = 1/b
                dw = 1/a/ceq
                pMat[j, i, 2:4] = [ceq, dw]
            except:
                pass
            nMat[j, i] = len(x)
        print('\t {} {} {}/{} {:.2f}'.format(i, codeC,
                                             j, nSite, time.time()-t0), end='\r')
print('total time {:.2f}'.format(time.time()-t0))

# better save as csv
df = pd.DataFrame(data=pMat[:, :, 0], index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'slope_a'))
df = pd.DataFrame(data=pMat[:, :, 1], index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'slope_b'))
df = pd.DataFrame(data=pMat[:, :, 2], index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'kate_ceq'))
df = pd.DataFrame(data=pMat[:, :, 3], index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'kate_dw'))
df = pd.DataFrame(data=nMat, index=siteNoLst, columns=codeLst)
df.index.name = 'siteNo'
df.to_csv(os.path.join(dirCQ, 'nSample'))
print('total time {:.2f}'.format(time.time()-t0))

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
