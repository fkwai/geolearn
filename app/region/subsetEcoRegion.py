
import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import grid
from hydroDL.post import plot
import numpy as np
import matplotlib.pyplot as plt


rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
lat, lon = df.getGeo()
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False).astype(int)

# print code list
ngrid = len(codeLst)
u1Lst, c1Lst = np.unique(codeLst[:, 0], return_counts=True)
for (u1, c1) in zip(u1Lst, c1Lst):
    # print('{:02d}-xx-xx {:.2f}% {:d}'.format(u1, c1/ngrid*100, c1))
    ind = np.where(codeLst[:, 0] == u1)[0]
    u2Lst, c2Lst = np.unique(codeLst[ind, 1], return_counts=True)
    for (u2, c2) in zip(u2Lst, c2Lst):
        # print('{:02d}-{:02d}-xx {:.2f}% {:d}'.format(u1, u2, c2/ngrid*100, c2))
        ind = np.where((codeLst[:, 0] == u1) & (codeLst[:, 1] == u2))[0]
        u3Lst, c3Lst = np.unique(codeLst[ind, 2], return_counts=True)
        for (u3, c3) in zip(u3Lst, c3Lst):
            print('{:02d}-{:02d}-{:02d} {:.2f}% {:d}'.format(u1,
                                                             u2, u3, c3/ngrid*100, c3))

# plot code maps


def indReg(l1, l2, l3):
    legStr = str(l1).zfill(2)
    if l2 == 0:
        ind = np.where((codeLst[:, 0] == l1))[0]
    else:
        legStr = legStr
        if l3 == 0:
            ind = np.where((codeLst[:, 0] == l1) & (
                codeLst[:, 1] == l2))[0]
        else:
            ind = np.where((codeLst[:, 0] == l1) & (
                codeLst[:, 1] == l2) & (codeLst[:, 2] == l3))[0]
    return ind, legStr


def codeReg(regLst):
    data = np.zeros(lat.shape)
    legLst = list()
    indLst = list()
    for (k, reg) in zip(range(len(regLst)), regLst):
        ind, legStr = indReg(reg[0], reg[1], reg[2])
        data[ind] = k+1
        legLst.append(legStr)
        indLst.append(ind)
    return data, legLst, indLst


regLst = [
    [9, 2, 3]
]
fig, ax = plt.subplots(figsize=(8, 6))
data, legLst, indLst = codeReg(regLst)
plot.plotMap(data, lat=lat, lon=lon, ax=ax, cRange=[0, len(legLst)])
fig.show()
