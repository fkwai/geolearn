
import matplotlib.dates as mdates
import random
import scipy
import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import json
import os
import importlib
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS
import matplotlib
import matplotlib.gridspec as gridspec


DF = dbBasin.DataFrameBasin('G200')
codeLst = usgs.varC


# LSTM corr
ep = 1000
dataName = 'G200'
trainSet = 'rmYr5'
testSet = 'pkYr5'
label = 'QFPRT2C'
outName = '{}-{}-{}'.format(dataName, label, trainSet)

# load TS
DF = dbBasin.DataFrameBasin(dataName)
yP, ycP = basinFull.testModel(outName, DF=DF, testSet='all', ep=1000)
codeLst = usgs.varC
# WRTDS
dirRoot = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format('G200N', trainSet, 'all')
yW = np.load(os.path.join(dirRoot, fileName)+'.npz')['arr_0']

# correlation
yPC=yP.copy()
yWC=yW.copy()
matNan = np.isnan(yP) | np.isnan(yW)
yPC[matNan] = np.nan
yWC[matNan] = np.nan
matObs = DF.c
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
yP1 = DF.extractSubset(yPC, trainSet)
yP2 = DF.extractSubset(yPC, testSet)
yW1 = DF.extractSubset(yWC, trainSet)
yW2 = DF.extractSubset(yWC, testSet)
importlib.reload(utils.stat)

statStr='Corr'
func=getattr(utils.stat,'cal'+statStr)
statL1=func(yP1,obs1)
statL2=func(yP2,obs2)
statW1=func(yW1,obs1)
statW2=func(yW2,obs2)

# count
matB = (~np.isnan(DF.c)*~np.isnan(DF.q[:, :, 0:1])
        ).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) & (count2 < 20)
for stat in [statL1, statL2, statW1, statW2]:
    stat[matRm] = np.nan
# load linear/seasonal
dictLR=dict()
for par in ['Q','S','QS']:
    dirPar = r'C:\Users\geofk\work\waterQuality\modelStat\LR-All\{}\param'.format(par)
    print(dirPar)
    matLR = np.full([len(DF.siteNoLst), len(codeLst)], np.nan)
    for k, code in enumerate(codeLst):
        filePar = os.path.join(dirPar, code)
        dfCorr = pd.read_csv(filePar, dtype={'siteNo': str}).set_index('siteNo')
        matLR[:, k] = dfCorr['rsq'].values
    matLR[matRm] = np.nan
    dictLR[par]=matLR



outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'

dictPlot = dict()
dictPlot['00010'] = ['01389500', '01667500', '09217000']
dictPlot['00300'] = ['01184490', '01616500', '02097464']
dictPlot['00915'] = ['12323600', '07047942', '06306250']
dictPlot['00618'] = ['01193500', '03497300', '06630000']

# selection plot
codePlotLst=['00010','00300','00915','00618']
codeStrLstTemp=[usgs.codePdf.loc[code]['shortName'] for code in codePlotLst]
codeStrLst=usgs.codeStrPlot(codeStrLstTemp)
figM = plt.figure(figsize=(12, 12))
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 1})
matplotlib.rcParams.update({'lines.markersize': 8})
import string
for k,code in enumerate(codePlotLst):
    j,i=utils.index2d(k,2,2)
    indC = codeLst.index(code)
    indS = np.where(~matRm[:, indC])[0]
    d1=count1[indS,indC]
    d2=count2[indS,indC]
    ind1 = np.where((d1<np.percentile(d1,60)) & (d1>np.percentile(d1,40)))[0]
    ind2 = np.where((d2<np.percentile(d2,60)) & (d2>np.percentile(d2,40)))[0]
    print(type(ind1))
    print(type(ind2))
    print(ind2)
    ind=np.intersect1d(ind1, ind2)
    x = statL2[indS, indC]
    y = statW2[indS, indC]
    xx=np.argsort(np.argsort(x))/x.shape[0]
    yy=np.argsort(np.argsort(y))/y.shape[0]
    gsM = gridspec.GridSpec(2, 2)
    axS = figM.add_subplot(gsM[j,i])    
    cs0 = axplot.scatter121(axS,xx,yy, d1,vR=[0,500],alpha=0.3)
    cs = axplot.scatter121(axS, xx[ind],yy[ind], d1[ind],vR=[0,500],edgecolors='black')
    siteNoLstCode=[DF.siteNoLst[jj] for jj in indS]
    for kk in [siteNoLstCode.index(siteNo) for siteNo in dictPlot[code]]:
        circle = plt.Circle([xx[kk], yy[kk]],0.05, color='r', fill=False)
        axS.add_patch(circle)
    axplot.titleInner(axS,'{}) {}'.format(string.ascii_uppercase[k],codeStrLst[k]))
    axS.set_aspect('equal')
    plt.colorbar(cs, orientation='vertical')
# figM.show()
# figM.savefig(os.path.join(outFolder, 'tsSel'))
# figM.savefig(os.path.join(outFolder, 'tsSel.svg'))

# figM.show()
# figM.savefig(os.path.join(outFolder, 'tsSel-leg.svg'))

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 8})

importlib.reload(figplot)
importlib.reload(axplot)
importlib.reload(mapplot)
codePlotLst=['00010','00300','00915','00618']
strLRLst=['S','S','Q','QS']
dictLRLabel={'S': 'seasonality',
             'Q': 'linearity',             
             'QS': 'simplicity'}


unitLst=[u'[\u2103]','[mg/L]','[mg/L]','[mg/L]']
for code,codeStr,strLR,unit in zip(codePlotLst,codeStrLst,strLRLst,unitLst):
    siteLst = dictPlot[code]    
    matLR=dictLR[strLR]
    
    outFolder = r'C:\Users\geofk\work\waterQuality\paper\G200'
    saveFolder = os.path.join(outFolder, code)
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    # ts map
    lat, lon = DF.getGeo()
    indC = codeLst.index(code)
    indS = np.where(~matRm[:, indC])[0]
    yrLst = np.arange(1985, 2020, 5).tolist()
    ny = len(yrLst)

    # plot map and scatter
    figM = plt.figure(figsize=(18, 3))
    gsM = gridspec.GridSpec(1, 5)
    axS = figM.add_subplot(gsM[0, :1])
    axS.set_title('A) LSTM vs WRTDS')
    cs = axplot.scatter121(axS, statL2[indS, indC],statW2[indS, indC], matLR[indS, indC],
                           size=25,alpha=0.5)
    axS.set_xlabel(r'$R_{LSTM}$')
    axS.set_ylabel(r'$R_{WRTDS}$')
    axS.set_aspect('equal')
    axS.tick_params(labelsize=12)
    ylimS1,ylimS2=axS.get_ylim()    
    cbar=plt.colorbar(cs, orientation='vertical', label=dictLRLabel[strLR])
    cbar.ax.tick_params(labelsize=12)
    for ind,letter in zip([DF.siteNoLst.index(siteNo) for siteNo in siteLst],'DEF'):        
        circle = plt.Circle([statL2[ind, indC], statW2[ind, indC]],
                            (ylimS2-ylimS1)/30, color='r', fill=False)
        axS.add_patch(circle)
        axS.text(statL2[ind, indC], statW2[ind, indC],letter)
    axM1,cb1 = mapplot.mapPoint(
        figM, gsM[0, 1:3], lat[indS], lon[indS], statL2[indS, indC], 
        s=25,cmap='viridis',alpha=0.5,returnCB=True)
    axM1.set_title(r'B) $R_{LSTM}$'+' of {}'.format(codeStr))
    cb1.ax.tick_params(labelsize=12)
    ylimM1,ylimM2=axM1.get_ylim()
    axM2,cb2 = mapplot.mapPoint(
        figM, gsM[0, 3:], lat[indS], lon[indS], statL2[indS, indC]**2-statW2[indS, indC]**2, 
        s=25,vRange=[-0.1, 0.1],cmap='viridis',alpha=0.5,returnCB=True)
    cb2.ax.tick_params(labelsize=12)
    axM2.set_title(r'C) $\Delta R^2_{LSTM-WRTDS}$'+' of {}'.format(codeStr))
    for ind,letter in zip([DF.siteNoLst.index(siteNo) for siteNo in siteLst],'DEF'):
        circle = plt.Circle([lon[ind], lat[ind]],(ylimM2-ylimM1)/30, color='r', fill=False)
        axM1.add_patch(circle)
        axM1.text(lon[ind], lat[ind],letter)
        circle = plt.Circle([lon[ind], lat[ind]],(ylimM2-ylimM1)/30, color='r', fill=False)
        axM2.add_patch(circle)
        axM2.text(lon[ind], lat[ind],letter)
    figM.tight_layout()
    figM.show()
    figM.savefig(os.path.join(saveFolder, 'map_{}'.format(code)))
    figM.savefig(os.path.join(saveFolder, 'map_{}.svg'.format(code)))

    # plot TS
    for siteNo, figN in zip(siteLst, 'DEF'):
        importlib.reload(axplot)
        ind = DF.siteNoLst.index(siteNo)
        dataPlot = [yW[:, ind, indC], yP[:, ind, indC],
                    DF.c[:, ind, DF.varC.index(code)]]
        cLst = 'kbr'
        cLst=  ['#377eb8','#e41a1c','k']
        # legLst = [r'WRTDS $\rho$={:.2f}'.format(corrW2[ind, indC]),
        #           r'LSTM $\rho$={:.2f}'.format(corrL2[ind, indC]),
        #           '{} obs'.format(codeStr)]
        legLst = ['WRTDS', 'LSTM', 'Obs.']
        figP = plt.figure(figsize=(15, 3))
        gsP = gridspec.GridSpec(1, ny, wspace=0)
        axP0 = figP.add_subplot(gsP[0, 0])
        axPLst = [axP0]
        for k in range(1, ny):
            axP = figP.add_subplot(gsP[0, k], sharey=axP0)
            axPLst.append(axP)
        axP = np.array(axPLst)
        axplot.multiYrTS(axP,  yrLst, DF.t, dataPlot, cLst=cLst)
        for ax in axP:
            ax.set_xlabel('')
            ax.set_xticklabels('')
        axplot.titleInner(axP[0],'{}{}'.format(codeStr,unit))        
        # axP[0].set_ylabel('{}{}'.format(codeStr,unit))
        titleStr = r'{}) {} of site {} {}={:.2f}; {}={:.2f}'.format(
            figN, codeStr, DF.siteNoLst[ind], '$R_{LSTM}$', statL2[ind, indC], '$R_{WRTDS}$', statW2[ind, indC])
        figP.suptitle(titleStr)
        # figP.tight_layout()
        figP.show()
        figP.savefig(os.path.join(saveFolder, 'tsYr5_{}_{}'.format(code, siteNo)))
        figP.savefig(os.path.join(
            saveFolder, 'tsYr5_{}_{}.svg'.format(code, siteNo)))
