import importlib
from hydroDL.post import figplot
import matplotlib.pyplot as plt
from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII
from hydroDL.post import axplot

import torch
import os
import json
import numpy as np

sE = 50
nE = 100
master = basins.wrapMaster('temp10', 'first80', saveEpoch=sE,
                           nEpoch=nE, saveName='temp1', optQ=1)
basins.trainModelTS(master)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp1', optQ=1)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp2', optQ=2)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp3', optQ=3)
# basins.trainModelTS('temp10', 'first80', saveEpoch=sE,
#                     nEpoch=nE, saveName='temp4', optQ=4)

# load master
modelName = 'temp1'
modelFolder = os.path.join(kPath.dirWQ, 'model', modelName)
masterFile = os.path.join(modelFolder, 'master.json')
with open(masterFile, 'r') as fp:
    master = json.load(fp)
statFile = os.path.join(modelFolder, 'stat.json')
with open(statFile, 'r') as fp:
    dictStat = json.load(fp)
statX = dictStat['statX']
statXC = dictStat['statXC']
statY = dictStat['statY']
statYC = dictStat['statYC']


# load data
testset = 'first80'
wqData = waterQuality.DataModelWQ(master['dataName'])
dataLst, statLst = wqData.transIn(subset='first80', optQ=master['optQ'])

# load model
nEpoch = 100
modelFile = os.path.join(modelFolder, 'model_ep{}'.format(nEpoch))
model = torch.load(modelFile)
sizeLst = trainTS.getSize(dataLst)
dataLst = trainTS.dealNaN(dataLst, master['optNan'])
x = dataLst[0]
xc = dataLst[1]
ny = sizeLst[2]

# np.where(np.isnan(dataLst[0]))

# test model - point by point
yOut, ycOut = trainTS.testModel(model, x, xc, ny)
qP, cP = wqData.transOut(yOut, ycOut, statLst[2], statLst[3])
statMat = wqData.calStatC(cP, subset=testset)
obsLst = wqData.extractSubset(testset)
qT, cT = obsLst[2:]

# plot
# get location
info = wqData.info.loc[wqData.subset[testset].tolist()].reset_index()
siteNoLst = info.siteNo.unique()
dfCrd = gageII.readData(
    varLst=['LAT_GAGE', 'LNG_GAGE'], siteNoLst=siteNoLst)
lat = dfCrd['LAT_GAGE'].values
lon = dfCrd['LNG_GAGE'].values
# codeSel = ['00955', '00940', '00915']
codeSel = ['00010', '00095']
icLst = [wqData.varC.index(code) for code in codeSel]
codePdf = waterQuality.codePdf


def funcMap():
    figM, axM = plt.subplots(len(codeSel), 2, figsize=(8, 6))
    for k in range(len(codeSel)):
        ic = icLst[k]
        axplot.mapPoint(axM[k, 0], lat, lon, statMat[:, ic, 0], s=6)
        axplot.mapPoint(axM[k, 1], lat, lon, statMat[:, ic, 1], s=6)
    figP, axP = plt.subplots(len(codeSel), 1, figsize=(8, 6))
    return figM, axM, figP, axP, lon, lat


def funcPoint(iP, axP):
    siteNo = siteNoLst[iP]
    for j in range(len(codeSel)):
        indS = info[info['siteNo'] == siteNo].index.values
        axP[j].plot(cT[indS, j], cP[indS, j], '*')


importlib.reload(figplot)
figplot.clickMap(funcMap, funcPoint)
