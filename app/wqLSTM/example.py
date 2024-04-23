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


# explain of DataFrameBasin
# load Ca concentration
code = "00915"
indC = DF.varC.index(code)
matC = DF.c[:, indC, :]
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
    dataPlot = [
        matC[:, iP],
        matQ[:, iP],
        matF[:, iP],
    ]
    axT1 = axP.twinx()
    axT2 = axP.twinx()
    axT2.spines["right"].set_position(("outward", 60))
    axplot.multiTS([axP, axT1, axT2], DF.t, dataPlot)
    titleStr = "{} {:.2f} {:.2f}".format(DF.siteNoLst[iP])
    axplot.titleInner(axP, titleStr)


# figplot.clickMap(funcM, funcP)
