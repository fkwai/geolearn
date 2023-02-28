import pandas as pd
from hydroDL.data import usgs, gageII, gridMET, ntn, GLASS, transform, dbBasin
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from hydroDL.post import axplot, figplot, mapplot
from hydroDL import kPath, utils
import os
from hydroDL.master import basinFull
from hydroDL.app.waterQuality import WRTDS


dataName = 'NY5'
DF = dbBasin.DataFrameBasin(dataName)

label = 'QFT2C'
trainSet = 'rmYr5b0'
testSet = 'pkYr5b0'


# load model performance
outName = '{}-{}-{}'.format(dataName, label, trainSet)
ep = 180


dictMaster = basinFull.loadMaster(outName)
varY = dictMaster['varY']
matObs = DF.extractT(varY)
obs1 = DF.extractSubset(matObs, trainSet)
obs2 = DF.extractSubset(matObs, testSet)
# LSTM
yP1, ycP1 = basinFull.testModel(outName, DF=DF, testSet=trainSet, ep=ep)
yP2, ycP2 = basinFull.testModel(outName, DF=DF, testSet=testSet, ep=ep)
corrL1 = utils.stat.calCorr(yP1, obs1)
corrL2 = utils.stat.calCorr(yP2, obs2)
# WRTDS
varY_WRTDS = usgs.varC
dirWRTDS = os.path.join(kPath.dirWQ, 'modelStat', 'WRTDS-dbBasin')
fileName = '{}-{}-{}'.format(dataName, trainSet, 'all.npz')
yW = np.load(os.path.join(dirWRTDS, fileName))['yW']
yW1 = DF.extractSubset(yW, trainSet)
yW2 = DF.extractSubset(yW, testSet)
corrW1 = utils.stat.calCorr(yW1, obs1)
corrW2 = utils.stat.calCorr(yW2, obs2)
# count
matB = (~np.isnan(DF.c)).astype(int).astype(float)
matB1 = DF.extractSubset(matB, trainSet)
matB2 = DF.extractSubset(matB, testSet)
count1 = np.nansum(matB1, axis=0)
count2 = np.nansum(matB2, axis=0)
matRm = (count1 < 80) | (count2 < 20)
for corr in [corrL1, corrL2]:
    corr[matRm] = np.nan


# dates
indT1, indT2, indS, mask = DF.readSubset(trainSet)
t1 = DF.t[indT1:indT2]
indT1, indT2, indS, mask = DF.readSubset(testSet)
t2 = DF.t[indT1:indT2]
tpd1 = pd.to_datetime(t1)
day1 = tpd1.dayofyear
year1 = tpd1.year
tpd2 = pd.to_datetime(t2)
day2 = tpd1.dayofyear
year2 = tpd1.year
matQ = DF.extractT(['runoff'])
q1 = DF.extractSubset(matQ, trainSet)
q2 = DF.extractSubset(matQ, testSet)
lat, lon = DF.getGeo()


code = '00660'
indC = varY.index(code)
indCW = varY_WRTDS.index(code)
indS = np.where(~matRm[:, indC])[0]


def funcM():
    figM = plt.figure(figsize=(8, 6))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat[indS], lon[indS], corrL2[indS, indC])
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
            corrW1[k, indCW], corrW2[k, indCW],corrL1[k, indC], corrL2[k, indC]) 
    print(strP)


figplot.clickMap(funcM, funcP)



# figP = plt.figure(figsize=(15, 3))
# gsP = gridspec.GridSpec(2, 5)
# axP1 = figP.add_subplot(gsP[0, 3])
# axP2 = figP.add_subplot(gsP[1, 3])
# axQ1 = figP.add_subplot(gsP[0, 4])
# axQ2 = figP.add_subplot(gsP[1, 4])
# axPT1 = figP.add_subplot(gsP[0, :3])
# axPT2 = figP.add_subplot(gsP[1, :3])
# # axPLst = [axP0, axP1, axPT]

# iP = 10
# print(iP)
# k = indS[iP]
# dataP1 = [yW1[:, k, indCW], yP1[:, k, indC], obs1[:, k, indC]]
# dataP2 = [yW2[:, k, indCW], yP2[:, k, indC], obs2[:, k, indC]]
# leg1 = ['WRTDS {:.2f}'.format(corrW1[k, indCW]), 'LSTM {:.2f}'.format(corrL1[k, indCW])]
# leg2 = ['WRTDS {:.2f}'.format(corrW2[k, indCW]), 'LSTM {:.2f}'.format(corrL2[k, indCW])]
# axplot.plotTS(axPT1, t1, dataP1, cLst='rbk', styLst='--*', legLst=leg1)
# axplot.plotTS(axPT2, t2, dataP2, cLst='rbk', styLst='--*', legLst=leg2)
# scP1 = axP1.scatter(day1, obs1[:, k, indC], c=year1)
# scP2 = axP2.scatter(day2, obs2[:, k, indC], c=year2)
# scQ1 = axQ1.scatter(np.log(q1[:, k, 0]), obs1[:, k, indC], c=day1)
# scQ2 = axQ2.scatter(np.log(q2[:, k, 0]), obs2[:, k, indC], c=day2)
# figP.colorbar(scP1)
# figP.colorbar(scP2)
# figP.colorbar(scQ1)
# figP.colorbar(scQ2)

# figP.show()
