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

wqData = waterQuality.DataModelWQ('basinRef')
figFolder = os.path.join(kPath.dirWQ, 'basinRef')

# compare
nameLst = ['areaLT10', 'areaGT2500', 'eco0503', 'eco0902', 'nutr06', 'nutr08']
for name in nameLst:
    outLst = ['basinRef-first50-opt1',
              'basinRef-{}-F50-opt1'.format(name),
              'basinRef-first50-opt2',
              'basinRef-{}-F50-opt2'.format(name)]
    trainSet = name+'-F50'
    testSet = name+'-L50'
    errMatLst1, errMatLst2 = [list(), list()]
    for outName in outLst:
        p1, o1 = basins.testModel(outName, trainSet, wqData=wqData)
        p2, o2 = basins.testModel(outName, testSet, wqData=wqData)
        errMat1 = wqData.errBySite(p1, subset=trainSet)
        errMat2 = wqData.errBySite(p2, subset=testSet)
        errMatLst1.append(errMat1)
        errMatLst2.append(errMat2)
        master = basins.loadMaster(outName)
        varAll = master['varYC']

    codePdf = usgs.codePdf
    groupLst = codePdf.group.unique().tolist()
    for errMatLst, train in zip([errMatLst1, errMatLst2], ['train', 'test']):
        for group in groupLst:
            codeLst = codePdf[codePdf.group == group].index.tolist()
            indLst = [wqData.varC.index(code) for code in codeLst]
            labLst1 = [codePdf.loc[code]['shortName'] +
                       '\n'+code for code in codeLst]
            labLst2 = ['opt1-ref',  'opt1-'+name, 'opt2-ref', 'opt2-'+name]
            dataBox = list()
            for ic in indLst:
                temp = list()
                for errMat in errMatLst:
                    temp.append(errMat[:, ic, 1])
                dataBox.append(temp)
            title = '{} correlation of {}'.format(train, group)
            figName = 'box_{}_{}_{}'.format(name, group, train)
            fig = figplot.boxPlot(dataBox, label1=labLst1, label2=labLst2)
            fig.suptitle(title)
            fig.show()
            fig.savefig(os.path.join(figFolder, figName))
