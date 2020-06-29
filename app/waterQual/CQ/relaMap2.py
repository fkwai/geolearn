from hydroDL import kPath, utils
from hydroDL.app import waterQuality, wqRela
from hydroDL.data import gageII, usgs, gridMET
from hydroDL.master import basins
from hydroDL.post import axplot, figplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time
from astropy.timeseries import LombScargle
import scipy.signal as signal
import matplotlib.gridspec as gridspec


# pick out sites that are have relative large number of observations
dirInv = os.path.join(kPath.dirData, 'USGS', 'inventory')
fileSiteNo = os.path.join(dirInv, 'siteNoLst-1979')
siteNoLstAll = pd.read_csv(fileSiteNo, header=None, dtype=str)[0].tolist()
dfAll = pd.read_csv(os.path.join(dirInv, 'codeCount.csv'),
                    dtype={'siteNo': str}).set_index('siteNo')

# pick some sites
# codeLst = ['00915', '00940', '00955','00300']
codeLst = ['00915', '00945', '00955']
tempLst = list()
for code in codeLst:
    temp = dfAll[dfAll[code] > 200].index.tolist()
    tempLst.append(temp)
siteNoLst = tempLst[0]
for k in range(1, len(tempLst)):
    siteNoLst = list(set(siteNoLst).intersection(tempLst[k]))

# # cal dw
# code = codeLst[0]
# pMat2 = np.ndarray([len(siteNoLst), 2])
# pdfArea = gageII.readData(varLst=['DRAIN_SQKM'], siteNoLst=siteNoLst)
# unitConv = 0.3048**3*365*24*60*60/1000**2
# for k, siteNo in enumerate(siteNoLst):
#     area = pdfArea.loc[siteNo]['DRAIN_SQKM']
#     dfC = usgs.readSample(siteNo, codeLst=codeLst)
#     dfQ = usgs.readStreamflow(siteNo)
#     df = dfC.join(dfQ)
#     t = df.index.values
#     q = df['00060_00003'].values/area*unitConv
#     c = df[code].values
#     ceq, dw, y = wqRela.kateModel(q, c)
#     pMat2[k, 0] = ceq
#     pMat2[k, 1] = dw

dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values



def funcMap():
    figM, axM = plt.subplots(1, 1, figsize=(8, 6))
    axplot.mapPoint(axM, lat, lon, lon, s=12)
    figP = plt.figure(constrained_layout=True)
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=figP)
    axLst = [figP.add_subplot(spec[0, :])] +\
        [figP.add_subplot(spec[1, k]) for k in range(3)]
    axP = np.array(axLst)
    return figM, axM, figP, axP, lon, lat

def funcPoint(iP, axes):
    kA = 0
    siteNo = siteNoLst[iP]
    startDate = pd.datetime(1979, 1, 1)
    endDate = pd.datetime(2019, 12, 31)
    ctR = pd.date_range(startDate, endDate)
    dfData = pd.DataFrame({'date': ctR}).set_index('date')
    dfC = usgs.readSample(siteNo, codeLst=codeLst, startDate=startDate)
    dfQ = usgs.readStreamflow(siteNo, startDate=startDate)
    dfQ = dfQ.rename(columns={'00060_00003': '00060'})
    dfData = dfData.join(dfQ)
    dfData = dfData.join(dfC)

    # plot normalized time series
    ax = axes[kA]
    kA = kA+1
    t = dfData.index.values
    dfDataN = (dfData-dfData.mean())/dfData.std()
    varLst = dfData.columns.tolist()
    data = [dfDataN[var].values for var in varLst]
    legLst = ['streamflow']+[usgs.codePdf.loc[code]['shortName']
                             for code in codeLst]
    axplot.plotTS(ax, t, data, styLst='-***', cLst='krgb', legLst=legLst)

    # plot C-Q
    nc = len(codeLst)
    for k in range(nc):
        code = codeLst[k]
        q = dfData['00060']
        c = dfData[code]
        [q, c], ind = utils.rmNan([q, c])
        ax = axes[kA]
        kA = kA+1
        ax.plot(np.log(q), np.log(c), 'r*')


figplot.clickMap(funcMap, funcPoint)
