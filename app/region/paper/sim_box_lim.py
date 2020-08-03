import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL import master
from hydroDL.post import plot, stat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib
caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']
caseLabLst = ['8.3.5', '9.3.1', '9.3.3',
              '9.4.1', '9.4.2', '10.1.5', '10.2.4']
saveFolder = r'C:\Users\geofk\work\paper\SMAP-regional'
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

# test
tRange = [20160401, 20180401]
subsetPattern = 'ecoReg_{}_L{}_v2f1'
levLst = [3, 2, 1, 0]
rootDB = pathSMAP['DB_L3_NA']
dfC = dbCsv.DataframeCsv(rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
latC, lonC = dfC.getGeo()

errLstAll = list()
for case in caseLst:
    testName = subsetPattern.format(case, 3)
    errLst = list()
    for k in levLst:
        if k in [0, 1]:
            # subset = 'ecoReg_{}_L{}_v2f1'.format(case, k)
            subset = 'ecoReg_{}_L{}_sampleLin'.format(case, k)
        else:
            subset = subsetPattern.format(case, k)
        outName = subset + '_Forcing'
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionCase', outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        err = stat.statError(yp[:, :, 0], yt[:, :, 0])
        errLst.append(err)
    errLstAll.append(errLst)

# plot box
cLst = 'ygbr'
keyLst = ['RMSE', 'Corr']
for key in keyLst:
    dataBox = list()
    for errLst in errLstAll:
        temp = list()
        for err in errLst:
            temp.append(err[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox, caseLabLst, colorLst=cLst,
                          figsize=(12, 4), sharey=True)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'sim_box_lim_{}'.format(key))
    fig.savefig(saveFile)


# plot box
cLst = 'ygbr'
keyLst = ['Corr', 'RMSE']
dataBox = list()
for key in keyLst:
    temp = list()
    for err in errLst:
        temp.append(err[key])
    dataBox.append(temp)
fig = plot.plotBoxFig(dataBox,
                      '  ', ['level ' + str(x) for x in levLst],
                      title='boxplot for ' + case,
                      figsize=(8, 6),
                      colorLst=cLst,
                      sharey=False)
# plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
fig.show()
saveFile = os.path.join(saveFolder, 'legend_box'.format(case))
fig.savefig(saveFile)

fig.savefig(saveFile + '.eps')
