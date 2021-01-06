# initial
from hydroDL import pathSMAP, master
import os
from hydroDL.data import dbCsv
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

subsetLst = ['ecoRegion{0:0>2}_v2f1'.format(x) for x in range(1, 18)]
caseLst1 = ['Local', 'Global']
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'

# load data and calculate stat
statLst = list()
tRange = [20160401, 20180401]
for k in range(len(subsetLst)):
    testName = subsetLst[k]
    tempLst = list()
    for case1 in caseLst1:
        if case1 == 'Local':
            out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegion',
                               subsetLst[k] + '_Forcing')
        elif case1 == 'Global':
            out = os.path.join(pathSMAP['Out_L3_NA'], 'CONUSv2f1_Forcing')
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        temp = stat.statError(yp[:, :, 0], yt[:, :, 0])
        tempLst.append(temp)
    statLst.append(tempLst)


# plot box
keyLst = ['RMSE', 'Corr']
# ecoLst = ['{0:0>2}'.format(x) for x in range(1, 18)]
ecoLst = 'ABDEFGHIJKLMNOPQR'
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
for k in range(len(keyLst)):
    dataBox = list()
    key = keyLst[k]
    for ss in statLst:
        temp = list()
        for s in ss:
            temp.append(s[key])
        dataBox.append(temp)
    if key == 'RMSE':
        fig = plot.plotBoxFig(dataBox,
                              ecoLst,
                              caseLst1,
                              widths=0.5,
                              figsize=(12, 4))
    else:
        fig = plot.plotBoxFig(dataBox, ecoLst, widths=0.5, figsize=(12, 4))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'ecoRegion_box_' + key)
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
