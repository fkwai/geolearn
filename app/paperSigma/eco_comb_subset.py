import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import app
# from hydroDL.post import plot
import numpy as np
import matplotlib.pyplot as plt


rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
lat, lon = df.getGeo()
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False).astype(int)

caseLst1 = [['08-03-00', '08-04-00'],
            ['09-02-00', '09-03-00'],
            ['10-01-04', '10-01-05', '10-01-06',
                '10-01-07', '10-01-08', '10-02-00']
            ]
tempLst = [['08-03-01', '08-03-02', '08-03-03', '08-04-00'],
           ['08-03-04', '08-03-05', '08-03-06', '08-03-07', '08-03-08'],
           ['09-02-00'],
           ['09-03-00'],
           ['10-01-04', '10-01-05', '10-01-06', '10-01-07', '10-01-08'],
           ['10-02-00']
           ]
caseLst = caseLst1
for i in [0, 1]:
    for j in [2, 3]:
        for k in [4, 5]:
            temp = tempLst[i]+tempLst[j]+tempLst[k]
            caseLst.append(temp)

# get subset
indSel = set()
for k, case in enumerate(caseLst):
    indLst = list()
    print(k)
    for reg in case:
        codeReg = [int(x) for x in reg.split('-')]
        ind = app.ecoReg_ind(codeReg, codeLst)
        indLst = indLst+ind.tolist()
    indSel.update(indLst)
    df.subsetInit('ecoComb{}_v2f1'.format(k), ind=indLst)
indAll = set(df.indSub.tolist())
indTest = list(indAll.difference(indSel))
df.subsetInit('ecoCombTest_v2f1', ind=indTest)
