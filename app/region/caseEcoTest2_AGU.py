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
caseLst = [
    '080305', '090301', '090303', '090401', '090402', '100105', '100204'
]
saveFolder = r'C:\Users\geofk\OneDrive\Documents\Presentation\AGU2019'
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

# case = '090303'
for case in caseLst:
    testName = subsetPattern.format(case, 3)
    errLst = list()
    for k in levLst:
        if k in [0, 1]:
            # if k in [-1]:
            subset = 'ecoReg_{}_L{}_v2f1'.format(case, k)
        else:
            subset = subsetPattern.format(case, k)
        outName = subset + '_Forcing'
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionCase', outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        errLst.append(stat.statError(yp[:, :, 0], yt[:, :, 0]))

    # plot box
    cLst = 'ygbr'
    keyLst = ['RMSE', 'Corr']
    dataBox = list()
    for key in keyLst:
        temp = list()
        for err in errLst:
            temp.append(err[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox,
                          '  ',
                          figsize=(8, 6),
                          colorLst=cLst,
                          sharey=False)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'case_{}_box'.format(case))
    fig.savefig(saveFile)

# plot box
cLst = 'ygbr'
keyLst = ['Corr','RMSE']
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
fig.savefig(saveFile + '.eps')
