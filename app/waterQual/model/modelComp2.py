import matplotlib.gridspec as gridspec
import os
import time
import json
import numpy as np
import pandas as pd
from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.post import plot, axplot
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime as dt
from hydroDL.data import gageII


caseName = 'refBasins'
nEpoch = 500
modelLst = ['modelA', 'modelB', 'modelC']
doLst = list()
# doLst.append('calStat')
doLst.append('loadData')
# doLst.append('plotBox')
doLst.append('plotTsMap')

if 'calStat' in doLst:
    for modelStr in modelLst:
        dictData, info = waterQuality.loadInfo(caseName)
        modelFolder = os.path.join(kPath.dirWQ, modelStr, caseName)
        targetFile = os.path.join(modelFolder, 'target.csv')
        dfT = pd.read_csv(targetFile, dtype={'siteNo': str})
        outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
        dfP = pd.read_csv(outFile, dtype={'siteNo': str})
        siteNoLst = dictData['siteNoLst']
        varC = dictData['varC']
        nP = len(siteNoLst)
        nC = len(varC)
        matRho1 = np.ndarray([nP, nC])
        matRho2 = np.ndarray([nP, nC])
        matRmse1 = np.ndarray([nP, nC])
        matRmse2 = np.ndarray([nP, nC])
        matN1 = np.ndarray([nP, nC])
        matN2 = np.ndarray([nP, nC])
        for iS, siteNo in enumerate(siteNoLst):
            print(iS)
            for iC, var in enumerate(varC):
                obs = dfT[dfT['siteNo'] == siteNoLst[iS]][varC[iC]].values
                pred = dfP[dfP['siteNo'] == siteNoLst[iS]][varC[iC]].values
                bTrain = dfT[dfT['siteNo'] == siteNoLst[iS]
                             ]['train'].values.astype(bool)
                ind1 = np.where(~np.isnan(obs) & bTrain)[0]
                ind2 = np.where(~np.isnan(obs) & ~bTrain)[0]
                matRho1[iS, iC] = np.corrcoef(obs[ind1], pred[ind1])[0, 1]
                matRho2[iS, iC] = np.corrcoef(obs[ind2], pred[ind2])[0, 1]
                matRmse1[iS, iC] = np.sqrt(np.mean((obs[ind1]-pred[ind1])**2))
                matRmse2[iS, iC] = np.sqrt(np.mean((obs[ind2]-pred[ind2])**2))
                matN1[iS, iC] = len(ind1)
                matN2[iS, iC] = len(ind2)
        saveFile = os.path.join(
            modelFolder, 'statResult_Ep{}.npz'.format(nEpoch))
        np.savez(saveFile, matRho1=matRho1, matRho2=matRho2,
                 matRmse1=matRmse1, matRmse2=matRmse2, matN1=matN1, matN2=matN2)

if 'loadData' in doLst:
    npfLst = list()
    dfPLst = list()
    for modelStr in modelLst:
        modelFolder = os.path.join(kPath.dirWQ, modelStr, caseName)
        saveFile = os.path.join(
            modelFolder, 'statResult_Ep{}.npz'.format(nEpoch))
        npf = np.load(saveFile)
        npfLst.append(npf)
        outFile = os.path.join(modelFolder, 'output_Ep' + str(nEpoch) + '.csv')
        dfP = pd.read_csv(outFile, dtype={'siteNo': str})
        dfPLst.append(dfP)
    targetFile = os.path.join(modelFolder, 'target.csv')
    dfT = pd.read_csv(targetFile, dtype={'siteNo': str})
    dictData, info = waterQuality.loadInfo(caseName)
    codePdf = waterQuality.codePdf
    grpLst = list(pd.unique(codePdf['group']))
    siteNoLst = dictData['siteNoLst']
    varC = dictData['varC']


if 'plotBox' in doLst:
    for grp in grpLst:
        grpCodeLst = codePdf[codePdf['group'] == grp].index.tolist()
        dataBox = list()
        for code in grpCodeLst:
            ind = dictData['varC'].index(code)
            # temp = [[npf['matRmse1'][:, ind], npf['matRmse2'][:, ind]] for npf in npfLst]
            temp = [[npf['matRho1'][:, ind], npf['matRho2'][:, ind]]
                    for npf in npfLst]
            dataBox.append([item for sublist in temp for item in sublist])
        legLst = [item for sublist in [[s+'_train', s+'_test']
                                       for s in modelLst] for item in sublist]
        labLst = codePdf[codePdf['group'] == grp]['shortName'].tolist()
        fig = plot.plotBoxFig(dataBox, label1=labLst,
                              label2=legLst, sharey=False)
        fig.show()

if 'plotTsMap' in doLst:
    # plot map
    iCLst = [0, 11]
    tempLst = [npfLst[0]['matRmse2'][:, iC] for iC in iCLst]
    temp = np.sum(tempLst, axis=0)

    indG = np.where(~np.isnan(temp))[0].tolist()
    npf = npfLst[0]
    dataLst = [npf['matRmse2'][indG, iC] for iC in iCLst]
    dataNLst = [npf['matN2'][indG, iC] for iC in iCLst]
    mapTitleLst = ['RMSE of ' + codePdf['shortName'][varC[iC]]
                   for iC in iCLst]
    siteNoLstTemp = [siteNoLst[i] for i in indG]
    dfCrd = gageII.readData(
        varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLstTemp)
    lat = dfCrd['LAT_GAGE'].values
    lon = dfCrd['LNG_GAGE'].values
    nTs = len(iCLst)
    nMap = len(dataLst)
    gsR = nTs
    figsize = [12, 8]
    # setup axes
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(gsR + nTs, nMap)
    gs.update(wspace=0.025, hspace=0.5)
    axTsLst = list()
    for k in range(nTs):
        axTs = fig.add_subplot(gs[k + gsR, :])
        axTsLst.append(axTs)
    for k in range(nMap):
        ax = fig.add_subplot(gs[0:gsR, k])
        axplot.plotMap(
            ax, lat, lon, dataLst[k], dataNLst[k], title=mapTitleLst[k], vRange=vRange)
        # plot ts

        def onclick(event):
            xClick = event.xdata
            yClick = event.ydata
            d = np.sqrt((xClick - lon)**2 + (yClick - lat)**2)
            iS = np.argmin(d)
            siteNo = siteNoLstTemp[iS]
            for ix in range(nTs):
                var = varC[iCLst[ix]]
                titleStr = '{} [{}] siteno {}'.format(
                    codePdf['fullName'][var], codePdf['unit'][var], siteNo,)
                obs = dfT[dfT['siteNo'] == siteNo][var].values
                print([data[iS] for data in dataLst])
                ax = axTsLst[ix]
                ax.clear()
                t = pd.to_datetime(dfT[dfT['siteNo'] == siteNo]['date']).values
                bTrain = dfT[dfT['siteNo'] ==
                             siteNo]['train'].values.astype(bool)
                ind = np.where(~np.isnan(obs))[0]
                ind1 = np.where(~np.isnan(obs) & bTrain)[0]
                ind2 = np.where(~np.isnan(obs) & ~bTrain)[0]
                if len(ind2) != 0:
                    tBar = t[ind1[-1]]+(t[ind2[0]]-t[ind1[-1]])/2
                else:
                    tBar = None
                tt = t[ind]
                y = list()
                for dfP in dfPLst:
                    y.append(dfP[dfP['siteNo'] == siteNo][var].values[ind])
                y.append(obs[ind])
                axplot.plotTS(ax, tt, y, tBar=tBar, cLst='rbgk',
                              legLst=modelLst+['obs'])
                ax.set_title(titleStr)
            plt.draw()
        fig.canvas.mpl_connect('button_press_event', onclick)
        # plt.tight_layout()
    fig.show()

pass
