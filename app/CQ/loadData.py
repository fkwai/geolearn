from hydroDL.data import dbBasin, usgs, gageII, gridMET, GLASS
import os
from hydroDL import kPath
import numpy as np
import pandas as pd
from hydroDL.post import axplot, figplot, mapplot
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime as dt

code = '00955'
dataName = '{}-B200'.format(code)
DF = dbBasin.DataFrameBasin(dataName)


data = DF.c[:, -2, 0]
np.nanmean(data)
siteNo = DF.siteNoLst[-2]

dfC, dfCF = usgs.readSample(siteNo=siteNo, codeLst=usgs.varC, flag=2)
sd = dt.datetime(1979, 1, 1)
ed = dt.datetime(2022, 12, 31)
t = pd.date_range(sd, ed)
codeLst = sorted(usgs.varC)
dfCount = pd.DataFrame(index=t, columns=codeLst).fillna(False)
dfCount.update(~dfC.isna())

dfCount.sum()


saveFile = os.path.join(kPath.dirUsgs, 'siteSel', 'matBool.npz')
npz = np.load(saveFile)
matC = npz['matC']
matF = npz['matF']
matQ = npz['matQ']
t = npz['t']
codeLst = list(npz['codeLst'])
siteNoLst = list(npz['siteNoLst'])

indS = siteNoLst.index(siteNo)
indC = codeLst.index(code)
np.sum(matC[:, indS, indC])

matB = matC * ~matF * matQ[:, :, None]
matCount = np.sum(matB, axis=0)
the = 200
matPick = matCount[:, 2:] > the

# tsmap of C-Q and C-T

meanC = np.nanmean(DF.c, axis=0)
lat, lon = DF.getGeo()


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, corrL2[indS, indC])
    axM.set_title('{} {}'.format(usgs.codePdf.loc[code]['shortName'], code))
    figP = plt.figure(figsize=(15, 3))
    gsP = gridspec.GridSpec(2, 5)
    axP1 = figP.add_subplot(gsP[0, 3])
    axP2 = figP.add_subplot(gsP[1, 3])
    axQ1 = figP.add_subplot(gsP[0, 4])
    axQ2 = figP.add_subplot(gsP[1, 4])
    axPT1 = figP.add_subplot(gsP[0, :3])
    axPT2 = figP.add_subplot(gsP[1, :3])
    axPLst = [axP1, axP2, axQ1, axQ2, axPT1, axPT2]
    axP = np.array(axPLst)
    return figM, axM, figP, axP, lon[indS], lat[indS]


def funcP(iP, axP):
    print(iP)
    k = indS[iP]
    [axP1, axP2, axQ1, axQ2, axPT1, axPT2] = axP
    dataP1 = [yW1[:, k, indCW], yP1[:, k, indC], obs1[:, k, indC]]
    dataP2 = [yW2[:, k, indCW], yP2[:, k, indC], obs2[:, k, indC]]
    axplot.plotTS(axPT1, t1, dataP1, cLst='rbk', styLst='--*')
    axplot.plotTS(axPT2, t2, dataP2, cLst='rbk', styLst='--*')
    scP1 = axP1.scatter(day1, obs1[:, k, indC], c=year1)
    scP2 = axP2.scatter(day2, obs2[:, k, indC], c=year2)
    scQ1 = axQ1.scatter(np.log(q1[:, k, 0]), obs1[:, k, indC], c=day1)
    scQ2 = axQ2.scatter(np.log(q2[:, k, 0]), obs2[:, k, indC], c=day2)
    strP = 'WRTDS {:.2f} {:.2f}; LSTM {:.2f} {:.2f}'.format(
        corrW1[k, indCW], corrW2[k, indCW], corrL1[k, indC], corrL2[k, indC]
    )
    print(strP)


figplot.clickMap(funcM, funcP)
