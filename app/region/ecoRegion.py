# %% initial
from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
import numpy as np

subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst1 = ['Local', 'CONUS']
caseLst2 = ['Forcing', 'Soilm']
saveFolder = os.path.join(pathSMAP['dirResult'], 'regionalization')

# %% load data and calculate stat
statLst = list()
tRange = [20160401, 20180401]
for k in range(len(subsetLst)):
    testName = subsetLst[k]
    tempLst = list()
    for case1 in caseLst1:
        for case2 in caseLst2:
            if case1 == 'Local':
                out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                                   subsetLst[k] + '_' + case2)
            elif case1 == 'CONUS':
                out = os.path.join(pathSMAP['Out_L3_NA'], 'CONUSv2f1_' + case2)
            df, yp, yt = master.test(out, tRange=tRange, subset=testName)
            temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
            tempLst.append(temp)
    statLst.append(tempLst)

# %% plot box
keyLst = stat.keyLst
caseLst = list()
for case1 in caseLst1:
    for case2 in caseLst2:
        caseLst.append(case1 + ' ' + case2)
ecoLst = ['{0:0>2}'.format(x) for x in range(1, 18)]
for k in range(len(keyLst)):
    dataBox = list()
    key = keyLst[k]
    for ss in statLst:
        temp = list()
        for s in ss:
            temp.append(s[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox, ecoLst, caseLst, title=key, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'ecoRegion_box_' + key)
    fig.savefig(saveFile)

# %% improvement from model vs CONUS
keyLst = ['RMSE', 'ubRMSE', 'Corr']
fig, axes = plt.subplots(1, len(keyLst),figsize=(18,6))
for kk in range(len(keyLst)):
    key=keyLst[kk]
    px = list()
    py = list()
    for ss in statLst:
        a = ss[0][key]
        b = ss[1][key]
        c = ss[2][key]
        px.append(np.nanmean((b-a)/a))
        py.append(np.nanmean((c-a)/a))
        # px=px+((b-a)/a).tolist()
        # py = py+((c-a)/a).tolist()
    plot.plotVS(px, py, ax=axes[kk], title=key,xlabel='improve from CONUS',ylabel='improve from Model')
    dist = np.square(px-np.mean(px))+np.square(py-np.mean(py))
    ind=np.argsort(-dist)[:4]
    for k in ind:
        axes[kk].text(px[k], py[k], '{:02d}'.format(k+1))
saveFile = os.path.join(saveFolder, 'ecoRegion_vs')
fig.savefig(saveFile)
fig.show()
