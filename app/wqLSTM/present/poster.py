
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib

dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yPA, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC

# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yWA = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

yP=yPA.copy()
yW=yWA.copy()
# correlation
matNan = np.isnan(yP) | np.isnan(yW)
yP[matNan] = np.nan
yW[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
yP1 = DF.extractSubset(yP, trainSet)
yP2 = DF.extractSubset(yP, testSet)
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
importlib.reload(utils.stat)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)

# box plot
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'svg.fonttype': 'none'})

rcParams =matplotlib.rcParams


# load linear/seasonal
dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\QS\param'
matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
for k, code in enumerate(codeLst):
    filePar = os.path.join(dirPar, code)
    dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
    matLR[:, k] = dfCorr['rsq'].values
matLR[matRm] = np.nan

saveFolder=r'C:\Users\geofk\work\Presentation\2023\poster'
# stat
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.gridspec as gridspec
statStr = 'KGE'
func=getattr(utils.stat,'cal'+statStr)
statL2=func(yP2,obs2)
statW2=func(yW2,obs2)

# map
code = '00945'
indC=codeLst.index(code)
codeStr = usgs.getCodeStr(code)
lat,lon=DF.getGeo()
gsM = gridspec.GridSpec(1, 1)

figM = plt.figure(figsize=(10, 4))
axM1,cb1 = mapplot.mapPoint(
    figM, gsM[0, 0], lat, lon, statL2[:, indC], 
    s=25,cmap='viridis',alpha=0.8,returnCB=True)
axplot.titleInner(axM1, r'$KGE_{LSTM}$'+' of {}'.format(codeStr))
# figM.show()
# figM.savefig(os.path.join(saveFolder, 'map_{}_{}.svg'.format(code, statStr)))

siteNo = '01349150'
ind = DF.siteNoLst.index(siteNo)
circle = plt.Circle([lon[ind], lat[ind]],1, color='r', fill=False)
axM1.add_patch(circle)
figM.show()
ind = DF.siteNoLst.index(siteNo)
 for ind,letter in zip([DF.siteNoLst.index(siteNo) for siteNo in siteLst],'DEF'):
        circle = plt.Circle([lon[ind], lat[ind]],(ylimM2-ylimM1)/30, color='r', fill=False)
        axM1.add_patch(circle)
        axM1.text(lon[ind], lat[ind],letter)
        circle = plt.Circle([lon[ind], lat[ind]],(ylimM2-ylimM1)/30, color='r', fill=False)

# ts
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 5})
matplotlib.rcParams.update({'svg.fonttype': 'none'})
siteNo = '01349150'
ind = DF.siteNoLst.index(siteNo)
dataPlot = [yWA[:, ind, indC], yPA[:, ind, indC],
            DF.c[:, ind, DF.varC.index(code)]]
cLst=  ['#377eb8','#e41a1c','k']
legLst = ['WRTDS', 'LSTM', 'Obs.']
figP = plt.figure(figsize=(15, 4))
yrLst = np.arange(1985, 2020, 5).tolist()
ny = len(yrLst)
gsP = gridspec.GridSpec(1, ny, wspace=0)
axP0 = figP.add_subplot(gsP[0, 0])
axPLst = [axP0]
for k in range(1, ny):
    axP = figP.add_subplot(gsP[0, k], sharey=axP0)
    axPLst.append(axP)
axP = np.array(axPLst)
axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst)
# for ax in axP:
#     ax.set_xlabel('')
#     # ax.set_xticklabels('')
axplot.titleInner(axP[0],'{}{}'.format(codeStr,'[mg/L]'))        
# axP[0].set_ylabel('{}{}'.format(codeStr,unit))
titleStr = r'{} of site {}'.format(codeStr, DF.siteNoLst[ind])
figP.suptitle(titleStr)
figP.tight_layout()
figP.show()
figP.savefig(os.path.join(saveFolder, 'ts_{}_{}.svg'.format(code, siteNo)))

fig,ax=plt.subplots(1,1)
ax.plot(DF.t,yP[:, ind, indC])
fig.show()

indPlot = np.argsort(np.nanmean(matLR, axis=0))

