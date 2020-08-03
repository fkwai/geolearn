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

rmseLst0 = list()
rmseLst1 = list()
corrLst0 = list()
corrLst1 = list()
for stat in statLst:
    rmseLst0 = rmseLst0+list(stat[0]['RMSE'])
    rmseLst1 = rmseLst1+list(stat[1]['RMSE'])
    corrLst0 = corrLst0+list(stat[0]['Corr'])
    corrLst1 = corrLst1+list(stat[1]['Corr'])
len(np.where(np.array(rmseLst0) > np.array(rmseLst1))[0])
len(np.where(np.array(corrLst0) < np.array(corrLst1))[0])


fig, axes = plt.subplots(1, 2)
x = rmseLst0
y = rmseLst1
vmin = np.nanmin([x, y])
vmax = np.nanmax([x, y])
axes[0].plot(x, y, '*')
axes[0].plot([vmin, vmax], [vmin, vmax], 'k-')
x = corrLst0
y = corrLst1
vmin = np.nanmin([x, y])
vmax = np.nanmax([x, y])
axes[1].plot(x, y, '*')
axes[1].plot([vmin, vmax], [vmin, vmax], 'k-')
fig.show()
