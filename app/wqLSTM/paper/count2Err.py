
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
DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC


# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

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
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)
nashL1 = utils.stat.calNash(yP1, obs1)
nashL2 = utils.stat.calNash(yP2, obs2)
nashW1 = utils.stat.calNash(yW1, obs1)
nashW2 = utils.stat.calNash(yW2, obs2)
kgeL1 = utils.stat.calKGE(yP1, obs1)
kgeL2 = utils.stat.calKGE(yP2, obs2)
kgeW1 = utils.stat.calKGE(yW1, obs1)
kgeW2 = utils.stat.calKGE(yW2, obs2)

# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2, corrW1, corrW2]:
    corr[matRm] = np.nan
for kge in [kgeL1, kgeL2, kgeW1, kgeW2]:
    kge[matRm] = np.nan
    kge[np.isinf(kge)] = np.nan
    kge[kge < -1] = np.nan
for nash in [nashL1, nashL2, nashW1, nashW2]:
    nash[matRm] = np.nan

import scipy.stats as stats
from matplotlib.ticker import LogLocator,LogFormatter

def generate_log_ticks(x_min, x_max):
    """Generate logarithmic major and minor ticks and labels within a given range."""
    # Major ticks: powers of 10 within the range
    start = int(np.ceil(x_min))
    end = int(np.floor(x_max))
    major_ticks = np.log10(np.logspace(start, end, end - start + 1))
    major_tick_labels = [f"{10**i:.0f}" for i in range(start, end + 1)]
    # Minor ticks: include numbers 2-9 for each power of 10
    minor_ticks = []
    for i in range(start, end):
        minor_ticks.extend(np.log10(np.linspace(2, 10, 9) * 10**i))

    return major_ticks, major_tick_labels, minor_ticks

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

for saveStr, errMat in zip(['corr', 'kge'], [corrL2, kgeL2]):
    fig, axes = plt.subplots(5, 4, figsize=(16, 10))
    for code in codeLst:
        ic = codeLst.index(code)
        ix, iy = utils.index2d(ic, 5, 4)
        c=count1[:,ic]
        c=np.log10(c)   
        r=errMat[:,ic]
        cc, rr = utils.rmNan([c,r], returnInd=False)
        c1, c2 = np.nanmin(cc), np.nanmax(cc)
        r1, r2 = np.nanmin(rr), np.nanmax(rr)
        # c1, c2 = 0, 1500
        # r1, r2 = 0, 1
        xx = np.linspace(c1, c2, 100)
        yy = np.linspace(r1, r2, 100)
        xm, ym = np.meshgrid(xx, yy)
        p = np.vstack([xm.ravel(), ym.ravel()])
        # k = stats.gaussian_kde([cc, rr],bw_method=0.5)
        k = stats.gaussian_kde([cc, rr])
        z = np.reshape(k(p).T, xm.shape)    
        # z=np.log10(z)
        
        # find medians
        # pLst = list()
        # xLevel=np.array([np.log10(x) for x in range(0,1500,100)])
        # for l1,l2 in zip(xLevel[:-1],xLevel[1:]):
        #     ind=np.where((cc>=l1)&(cc<l2))[0]
        #     pLst.append(np.nanmedian(rr[ind]))
        # xLst=(xLevel[:-1]+xLevel[1:])/2
        # x_med,p_med=utils.rmNan([xLst,np.array(pLst)],returnInd=False)
        
        levels=np.percentile(z,[25,50,75,85,90,95,99,100])
        ax=axes[ix,iy]
        ax.contourf(xx, yy, z,levels=levels, cmap='viridis')
        # axes[ix,iy].contourf(xx, yy, z, cmap='viridis')
        ax.plot(cc, rr, '.',markersize=1.5,color='#e41a1c') 
        # ax.plot(x_med,p_med,'k-')
        major_ticks, major_tick_labels, minor_ticks = generate_log_ticks(c1,c2)   
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_tick_labels)
        ax.set_xticks(minor_ticks, minor=True)
        titleStr='{}'.format(usgs.codePdf.loc[code]['shortName'])
        codeStr = usgs.codePdf.loc[code]['shortName']
        if codeStr in usgs.dictLabel.keys():
            titleStr=usgs.dictLabel[codeStr]
        else:
            titleStr=codeStr
        ax.text(0.95, 0.05, titleStr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right')
    fig.show()
    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    plt.savefig(os.path.join(dirPaper, 'count-{}-all'.format(saveStr)))
    plt.savefig(os.path.join(dirPaper, 'count-{}-all.svg'.format(saveStr)))


# 4 plots
codePlot=['00300','00915','00618','00945']

for saveStr, errMat in zip(['corr', 'kge'], [corrL2, kgeL2]):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for kk,code in enumerate(codePlot):
        ic = codeLst.index(code)
        ix, iy = utils.index2d(kk, 2, 2)
        c=count1[:,ic]
        c=np.log10(c)   
        r=errMat[:,ic]
        cc, rr = utils.rmNan([c,r], returnInd=False)
        c1, c2 = np.nanmin(cc), np.nanmax(cc)
        r1, r2 = np.nanmin(rr), np.nanmax(rr)        
        xx = np.linspace(c1, c2, 100)
        yy = np.linspace(r1, r2, 100)
        xm, ym = np.meshgrid(xx, yy)
        p = np.vstack([xm.ravel(), ym.ravel()])
        k = stats.gaussian_kde([cc, rr])
        z = np.reshape(k(p).T, xm.shape)    
        levels=np.percentile(z,[25,50,75,85,90,95,99,100])
        ax=axes[ix,iy]
        
        # find medians
        # pLst = list()
        # xLevel=np.array([np.log10(x) for x in range(0,1500,100)])
        # for l1,l2 in zip(xLevel[:-1],xLevel[1:]):
        #     ind=np.where((cc>=l1)&(cc<l2))[0]
        #     pLst.append(np.nanmedian(rr[ind]))
        # xLst=(xLevel[:-1]+xLevel[1:])/2
        # x_med,p_med=utils.rmNan([xLst,np.array(pLst)],returnInd=False)

        ax.contourf(xx, yy, z,levels=levels, cmap='viridis')
        # axes[ix,iy].contourf(xx, yy, z, cmap='viridis')
        ax.plot(cc, rr, '.',markersize=3,color='#e41a1c') 
        # ax.plot(x_med,p_med,'k-')
        major_ticks, major_tick_labels, minor_ticks = generate_log_ticks(c1,c2)   
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_tick_labels)
        ax.set_xticks(minor_ticks, minor=True)
        titleStr='{}'.format(usgs.codePdf.loc[code]['shortName'])
        codeStr = usgs.codePdf.loc[code]['shortName']
        if codeStr in usgs.dictLabel.keys():
            titleStr=usgs.dictLabel[codeStr]
        else:
            titleStr=codeStr
        ax.text(0.95, 0.05, titleStr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right')
    fig.show()
    dirPaper = r'C:\Users\geofk\work\waterQuality\paper\G200'
    plt.savefig(os.path.join(dirPaper, 'count-{}'.format(saveStr)))
    plt.savefig(os.path.join(dirPaper, 'count-{}.svg'.format(saveStr)))

# legend
x = np.linspace(1, 100, 100)  # Sample x data
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(Y) * np.cos(X)
perc=[25,50,75,85,90,95,99,100]
fig, ax = plt.subplots()
CS = ax.contourf(np.log(X), Y, Z,levels=levels, cmap='viridis')
cbar = fig.colorbar(CS)
cbar.set_ticklabels(['{}%'.format(lev) for lev in perc])
cbar.ax.set_ylabel('KGE')
fig.show()
plt.savefig(os.path.join(dirPaper, 'count-{}-leg'.format(saveStr)))
plt.savefig(os.path.join(dirPaper, 'count-{}-leg.svg'.format(saveStr)))
