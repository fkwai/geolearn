
from hydroDL.post import axplot, figplot
from hydroDL import kPath, utils
from hydroDL.data import gageII, usgs, gridMET, dbBasin
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from hydroDL.master import basinFull


dataName = 'G200'
DF = dbBasin.DataFrameBasin(dataName)

# count for code
codeLst = ['00600', '00618', '00915', '00945', '00955']
pLst = [100, 75, 50, 25]
nyLst = [6, 8, 10]

# load data
dictY = dict()
for code in codeLst:
    for ny in nyLst:
        for p in pLst:
            label = 'QFPRT2C'
            trainSet = '{}-n{}-p{}-B10'.format(code, ny, p)
            testSet = '{}-n{}-A10'.format(code, ny)
            outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
            yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet)
            dictY[trainSet] = yP

# correlation
for code in codeLst:
    indC = DF.varC.index(code)
    dataBox = list()
    label1 = list()
    testSet = '{}-n{}-A10'.format(code, 10)
    for ny in nyLst:
        obs = DF.extractSubset(DF.c, testSet)[:, :, indC]
        temp = list()
        for p in pLst:
            trainSet = '{}-n{}-p{}-B10'.format(code, ny, p)
            outName = '{}-{}-{}-{}'.format(dataName, label, trainSet, code)
            yP, ycP = basinFull.testModel(outName, DF=DF, testSet=testSet)
            corr = utils.stat.calCorr(yP[:, :, 0], obs)
            temp.append(corr)
        dataBox.append(temp)
        label1.append('{} basins'.format(len(corr)))
    label2 = ['train with {}% obs'.format(p) for p in pLst]
    fig, axes = figplot.boxPlot(
        dataBox, widths=0.5, label1=label1, label2=label2)
    fig.suptitle('{} {}'.format(code, usgs.codePdf['shortName'][code]))
    fig.show()
    outFolder = r'C:\Users\geofk\work\temp\tempFig'
    fig.savefig(os.path.join(outFolder, code))
