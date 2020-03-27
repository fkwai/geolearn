from hydroDL.master import basins
from hydroDL.app import waterQuality
from hydroDL import kPath
from hydroDL.model import trainTS
from hydroDL.data import gageII, usgs
from hydroDL.post import axplot, figplot

import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt

wqData = waterQuality.DataModelWQ('HBN')
figFolder = os.path.join(kPath.dirWQ, 'HBN', 'years')

# compare of opt1-4
yrLst = ['80s', '90s', '00s', '10s']
for yr in yrLst:
    outLst = ['HBN-{}-rm-opt1'.format(yr), 'HBN-{}-rm-opt2'.format(yr)]
    trainSet = '{}-rm'.format(yr)
    testSet = yr
    # outLst = ['HBN-opt1', 'HBN-opt2',
    #           'HBN-opt3', 'HBN-opt4']
    # trainSet = 'first80'
    # testSet = 'last20'
    pLst1, pLst2, errMatLst1, errMatLst2 = [list() for x in range(4)]
    for outName in outLst:
        p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
        p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
        errMat1 = wqData.errBySite(p1, subset=trainSet)
        errMat2 = wqData.errBySite(p2, subset=testSet)
        pLst1.append(p1)
        pLst2.append(p2)
        errMatLst1.append(errMat1)
        errMatLst2.append(errMat2)

    codePdf = usgs.codePdf
    groupLst = codePdf.group.unique().tolist()
    for group in groupLst:
        codeLst = codePdf[codePdf.group == group].index.tolist()
        indLst = [wqData.varC.index(code) for code in codeLst]
        labLst1 = [codePdf.loc[code]['shortName'] +
                   '\n'+code for code in codeLst]
        labLst2 = ['opt1-train', 'opt2-train', 'opt1-test', 'opt2-test']
        dataBox = list()
        for ic in indLst:
            temp = list()
            for errMat in errMatLst1:
                temp.append(errMat[:, ic, 1])
            for errMat in errMatLst2:
                temp.append(errMat[:, ic, 1])
            dataBox.append(temp)
        title = 'correlation of {} group on {}'.format(group, yr)
        figName = 'box_{}_{}'.format(group, yr)
        fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
        fig.suptitle(title)
        fig.show()
        fig.savefig(os.path.join(figFolder, figName))
