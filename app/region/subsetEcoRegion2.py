import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.utils import grid
from hydroDL.data import dbCsv
from hydroDL.post import plot
from hydroDL.master import default, wrapMaster, runTrain, train
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib.pyplot as plt
import numpy as np

caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']

# init
rootDB = pathSMAP['DB_L3_NA']
tRange = [20150401, 20160401]
df = dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
subsetPattern = 'ecoRegionL3_{}_v2f1'
lat, lon = df.getGeo()
fieldLst = ['ecoRegionL'+str(x+1) for x in range(3)]
codeLst = df.getDataConst(fieldLst, doNorm=False, rmNan=False).astype(int)

# case = caseLst[2]
for case in caseLst:
    setL3 = set(ecoReg_ind(case, codeLst))
    setL2 = set(ecoReg_ind(case[0:4]+'00', codeLst))
    setL1 = set(ecoReg_ind(case[0:2]+'0000', codeLst))
    setL0 = set(df.indSub)

    indL3 = list(setL3)
    indL2 = list(setL2)

    indTemp = np.array(list(setL1-setL2))
    smp = np.linspace(0, len(setL1-setL2)-1, len(setL2-setL3))
    smpInd = np.unique(smp.round()).astype(int)
    indL1 = indTemp[smpInd].tolist()+indL3

    indTemp = np.array(list(setL0-setL1))
    smp = np.linspace(0, len(setL0-setL1)-1, len(setL2-setL3))
    smpInd = np.unique(smp.round()).astype(int)
    indL0 = indTemp[smpInd].tolist()+indL3

    # .union(setL3))
    # df.subsetInit('ecoReg_{}_L3_v2f1'.format(case), ind=indL3)
    # df.subsetInit('ecoReg_{}_L2_v2f1'.format(case), ind=indL2)
    df.subsetInit('ecoReg_{}_L1_sampleLin'.format(case), ind = indL1)
    df.subsetInit('ecoReg_{}_L0_sampleLin'.format(case), ind = indL0)

    data = np.tile(np.array(to_rgb('lightgrey')), (lat.shape[0], 1))
    data[indL0, :] = np.array(to_rgb('k'))
    data[indL1, :] = np.array(to_rgb('b'))
    data[indL2, :] = np.array(to_rgb('g'))
    data[indL3, :] = np.array(to_rgb('r'))
    fig, ax = plt.subplots(figsize=(8, 4))
    plot.plotMap(data, lat=lat, lon=lon, ax=ax,
                 cbar=False, title=case, plotPoint=True)
    fig.show()
