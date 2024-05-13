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


dataName = "G200"
DF = dbBasin.DataFrameBasin(dataName)


code = '80154'
siteLst=['08334000','08340500','08353000','09382000','09397300']
siteId='08334000'

indC = DF.varC.index(code)
for site in siteLst:
    indS = DF.siteNoLst.index(site)
    v = DF.c[:, indS, indC]
    t=DF.t
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, v,'*')
    ax.set_title(site)
    fig.show()


# explain of DataFrameBasin
# load Ca concentration
code = "00915"
indC = DF.varC.index(code)
matC = DF.c[:, :, indC]
# load runoff
varQ = "runoff"
matQ = DF.q[:, :, DF.varQ.index(varQ)]
# load precipitation
varF = "pr"
matF = DF.f[:, :, DF.varF.index(varF)]
# load basin area
varG = "DRAIN_SQKM"
matG = DF.g[:, DF.varG.index(varG)]

# plot
lat, lon = DF.getGeo()


def funcM():
    figM = plt.figure(figsize=(6, 4))
    gsM = gridspec.GridSpec(1, 1)
    axM = mapplot.mapPoint(figM, gsM[0, 0], lat, lon, matG)
    axM.set_title("basin area")
    figP, axP = plt.subplots(1, 1, figsize=(15, 3))
    return figM, axM, figP, axP, lon, lat


def funcP(iP, axP):
    print(iP)
    axT1 = axP.twinx()
    axT2 = axP.twinx()
    axT2.spines["right"].set_position(("outward", 60))
    axT2.invert_yaxis()
    axP.plot(DF.t, matC[:, iP],'r*',label=usgs.getCodeStr(code))
    axT1.plot(DF.t, matQ[:, iP],'b-',label='runoff')
    axT2.plot(DF.t, matF[:, iP],'c-',label='precipitation')
    titleStr = "{}".format(DF.siteNoLst[iP])
    axplot.titleInner(axP, titleStr)


figplot.clickMap(funcM, funcP)

# train LSTM model
dataName = 'G200'
label = 'QFPR2C'
# DF = dbBasin.DataFrameBasin(dataName)
rho = 365
nbatch = 500
hs = 256
trainSet = 'rmYr5'
testSet = 'pkYr5'
varX = dbBasin.label2var(label.split('2')[0])
mtdX = dbBasin.io.extractVarMtd(varX)
varY = dbBasin.label2var(label.split('2')[1])
mtdY = dbBasin.io.extractVarMtd(varY)
varXC = gageII.varLst
mtdXC = dbBasin.io.extractVarMtd(varXC)
varYC = None
mtdYC = dbBasin.io.extractVarMtd(varYC)
outName = '{}-{}-{}-hs{}'.format(dataName, label, trainSet, hs)
dictP = basinFull.wrapMaster(
    outName=outName, dataName=dataName, trainSet=trainSet,
    nEpoch=500, saveEpoch=50, crit='RmseLoss3D',
    varX=varX, varY=varY, varXC=varXC, varYC=varYC,
    mtdX=mtdX, mtdY=mtdY, mtdXC=mtdXC, mtdYC=mtdYC,
    hiddenSize=hs, batchSize=[rho, nbatch])
# cmdP = 'python /home/users/kuaifang/GitHUB/geolearn/hydroDL/master/cmd/basinFull.py -M {}'
# slurm.submitJobGPU(outName, cmdP.format(outName), nH=24, nM=64)
basinFull.trainModel(outName)