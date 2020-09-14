
from hydroDL import kPath, utils
from hydroDL.app import waterQuality
from hydroDL.master import basins
from hydroDL.data import usgs, gageII, gridMET, ntn
from hydroDL.master import slurm
from hydroDL.post import axplot, figplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


codeLst = ['00618', '00915', '00930', '00940',
           '00945', '00950', '70303', '71846']
theLst = [50, 100, 200, 250, 250, 1.5, 1000, 35]

ep = 500
reTest = False
dataName = 'sbWT'
wqData = waterQuality.DataModelWQ(dataName)

pMat = np.ndarray([len(codeLst), 4])
for k in range(len(codeLst)):
    code = codeLst[k]
    the = theLst[k]
    trainSet = '{}-Y1'.format(code)
    testSet = '{}-Y2'.format(code)
    outName = '{}-{}-{}-{}'.format(dataName, code, 'ntnq', trainSet)
    siteNoLst = wqData.info.iloc[wqData.subset[trainSet]].siteNo.unique()
    ic = wqData.varC.index(code)
    yP, ycP = basins.testModel(
        outName, testSet, wqData=wqData, ep=ep, reTest=reTest)
    ind = wqData.subset[testSet]
    bdate = wqData.info.iloc[ind]['date'].values > np.datetime64('1980-01-01')
    o = wqData.c[-1, ind[bdate], ic]
    p = yP[-1, bdate, 0]

    n1 = len(np.where(o > the)[0])
    n2 = len(np.where(o <= the)[0])
    if n1 != 0:
        p1 = len(np.where((o > the) & (p > the))[0])/n1  # true positive
        p2 = len(np.where((o > the) & (p <= the))[0])/n1  # false negative
    else:
        p1 = 0
        p2 = 0
    p3 = len(np.where((o <= the) & (p <= the))[0])/n2  # true negtive
    p4 = len(np.where((o <= the) & (p > the))[0])/n2  # false negative
    pMat[k, :] = [p1, p2, p3, p4]


nameLst = usgs.codePdf.loc[codeLst]['shortName'].values
columns = ['true positive', 'false negative', 'true negative', 'false positive']
df = pd.DataFrame(index=nameLst, columns=columns, data=pMat)
