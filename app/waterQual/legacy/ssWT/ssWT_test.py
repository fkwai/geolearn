from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import matplotlib.pyplot as plt
varNtnLst = ['ph', 'Conduc', 'Ca', 'Mg', 'K', 'Na', 'NH4', 'NO3', 'Cl', 'SO4']
varNtnUsgsLst = ['00400', '00095', '00915', '00925', '00935',
                 '00930', '71846', '00618', '00940', '00945']
codeLst = varNtnUsgsLst

ep = 500
reTest = False

corrMat = np.full([600, len(codeLst), 4], np.nan)
rmseMat = np.full([600, len(codeLst), 4], np.nan)

label = 'ntnS'

for iD, dataName in enumerate(['ssW', 'ssWT']):
    wqData = waterQuality.DataModelWQ(dataName)
    siteNoLst = wqData.info.siteNo.unique()
    for iCode, code in enumerate(codeLst):
        trainSet = '{}-Y1'.format(code)
        testSet = '{}-Y2'.format(code)
        outName = '{}-{}-{}-{}'.format(dataName, code, label, trainSet)
        master = basins.loadMaster(outName)
        ic = wqData.varC.index(code)
        for iT, subset in enumerate([trainSet, testSet]):
            yP, ycP = basins.testModel(
                outName, subset, wqData=wqData, ep=ep, reTest=reTest)
            ind = wqData.subset[subset]
            info = wqData.info.iloc[ind].reset_index()
            if dataName == 'ssWT':
                o = wqData.c[-1, ind, ic]
                p = yP[-1, :, 1]
            elif dataName == 'ssW':
                o = wqData.c[ind, ic]
                p = ycP[:, 0]
            for iS, siteNo in enumerate(siteNoLst):
                indS = info[info['siteNo'] == siteNo].index.values
                if len(indS) > 0:
                    [a, b], indV = utils.rmNan([o[indS], p[indS]])
                    corr = np.corrcoef(a, b)[0, 1]
                    rmse = np.sqrt(np.nanmean((a-b)**2))
                    corrMat[iS, iCode, iT+iD*2] = corr
                    rmseMat[iS, iCode, iT+iD*2] = rmse



# plot box
labLst1 = [usgs.codePdf.loc[code]['shortName'] +
           '\n'+code for code in codeLst]
labLst2 = ['train end', 'train seq','test end', 'test seq']
dataBox = list()
for k in range(len(codeLst)):
    code = codeLst[k]
    temp = list()
    for i in [0, 2, 1, 3]:
        temp.append(corrMat[:, k, i])
        # temp.append(rmseMat[:, k, i])
    dataBox.append(temp)
fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
                      label2=labLst2, figsize=(12, 4), yRange=[0, 1])
# fig = figplot.boxPlot(dataBox, label1=labLst1, widths=0.5,
#                       label2=labLst2, figsize=(12, 4), sharey=False)
fig.show()
