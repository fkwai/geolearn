from hydroDL import kPath
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt

dataName = 'ssW'
wqData = waterQuality.DataModelWQ(dataName)
caseLst = list()

varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
codeLst = varNtnUsgsLst

ep = 500
reTest = False
# single
errMatLst1 = list()
errMatLst2 = list()
for code in codeLst:
    labelLst = ['plain', 'ntnS']
    trainSet = '{}-Y1'.format(code)
    testSet = '{}-Y2'.format(code)
    errTemp1 = list()
    errTemp2 = list()
    for label in labelLst:
        outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        master = basins.loadMaster(outName)
        yP1, ycP1 = basins.testModel(
            outName, trainSet, wqData=wqData, ep=ep, reTest=reTest)
        yP2, ycP2 = basins.testModel(
            outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
        errMatC1 = wqData.errBySiteC(
            ycP1, subset=trainSet, varC=master['varYC'])
        errMatC2 = wqData.errBySiteC(
            ycP2, subset=testSet, varC=master['varYC'])
        errTemp1.append(errMatC1)
        errTemp2.append(errMatC2)
    errMatLst1.append(errTemp1)
    errMatLst2.append(errTemp2)

# comb
errCombLst1 = list()
errCombLst2 = list()
for code in codeLst:
    labelLst = ['ntnS']
    trainSet = 'comb-Y1'
    testSet = '{}-Y2'.format(code)
    errTemp1 = list()
    errTemp2 = list()
    for label in labelLst:
        outName = '{}-{}-{}-{}'.format(dataName, 'comb', label, trainSet)
        master = basins.loadMaster(outName)
        varComb = master['varYC']
        ic = varComb.index(code)
        yP1, ycP1 = basins.testModel(
            outName, trainSet, wqData=wqData, ep=ep, reTest=reTest)
        yP2, ycP2 = basins.testModel(
            outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
        errMatC1 = wqData.errBySiteC(
            ycP1[:, ic:ic+1], subset=trainSet, varC=[code])
        errMatC2 = wqData.errBySiteC(
            ycP2[:, ic:ic+1], subset=testSet, varC=[code])
        errTemp1.append(errMatC1)
        errTemp2.append(errMatC2)
    errCombLst1.append(errTemp1)
    errCombLst2.append(errTemp2)

# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train plain', 'train ntn', 'train comb ntn',
           'test plain', 'test ntn', 'test comb ntn']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    errLst1 = errMatLst1[k]
    errLst2 = errMatLst2[k]
    errComb1 = errCombLst1[k]
    errComb2 = errCombLst2[k]
    temp = list()
    # for errMat in errLst2+errComb2:
    for errMat in errLst1+errComb1+errLst2+errComb2:
        temp.append(errMat[:, 0, 1])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
