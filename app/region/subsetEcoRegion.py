
import hydroDL
from hydroDL.data import dbCsv
from hydroDL.utils import grid
from hydroDL.post import plot
import numpy as np

rootDB = hydroDL.pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUS', tRange=tRange)
lat, lon = df.getGeo()

# subset
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False).astype(int)

ngrid = len(codeLst)
u1Lst, c1Lst = np.unique(codeLst[:, 0], return_counts=True)
for (u1, c1) in zip(u1Lst, c1Lst):
    print('{:02d}-xx-xx {:.2f}% {:d}'.format(u1, c1/ngrid*100, c1))
    ind = np.where(codeLst[:, 0] == u1)[0]
    u2Lst, c2Lst = np.unique(codeLst[ind, 1], return_counts=True)
    for (u2, c2) in zip(u2Lst, c2Lst):
        print('{:02d}-{:02d}-xx {:.2f}% {:d}'.format(u1, u2, c2/ngrid*100, c2))
        ind = np.where((codeLst[:, 0] == u1) & (codeLst[:, 1] == u2))[0]
        u3Lst, c3Lst = np.unique(codeLst[ind, 2], return_counts=True)
        for (u3, c3) in zip(u3Lst, c3Lst):
            print('{:02d}-{:02d}-{:02d} {:.2f}% {:d}'.format(u1,
                                                             u2, u3, c3/ngrid*100, c3))


# dataGrid, uy, ux = grid.array2grid(codeLst[0], lat=lat, lon=lon)
# fig, ax = plot.plotMap(dataGrid, lat=uy, lon=ux)
