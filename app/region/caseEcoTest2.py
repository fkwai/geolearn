import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL import master
from hydroDL.post import plot, stat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

caseLst = ['080305', '090301', '090303',
           '090401', '090402', '100105', '100204']
saveFolder = os.path.join(pathSMAP['dirResult'], 'regionalization', 'case2')

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
            subset = 'ecoReg_{}_L{}_sampleLin'.format(case, k)
        else:
            subset = subsetPattern.format(case, k)
        outName = subset + '_Forcing'
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionCase',
                        outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        errLst.append(stat.statError(yp[:, :, 0], yt[:, :, 0]))

    # plot box
    cLst = 'rbgy'
    keyLst = stat.keyLst
    dataBox = list()
    for key in keyLst:
        temp = list()
        for err in errLst:
            temp.append(err[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox,
                        keyLst,
                        ['level '+str(x) for x in levLst],
                        title='boxplot for ' + case,
                        figsize=(12, 4), colorLst=cLst,
                        sharey=False)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'case_{}_sample_box'.format(case))
    fig.savefig(saveFile)

    # plot maps
    fig, ax = plt.subplots(figsize=(6, 3))
    data = np.tile(np.array(to_rgb('lightgrey')), (latC.shape[0], 1))
    for k, c in zip(reversed(levLst), reversed(cLst)):
        if k in [0, 1]:
        # if k in [-1]:
            subset = 'ecoReg_{}_L{}_sampleLin'.format(case, k)
        else:
            subset = subsetPattern.format(case, k)
        dfsub = dbCsv.DataframeCsv(
            rootDB=rootDB, subset=subset, tRange=tRange)
        data[dfsub.indSub, :] = np.array(to_rgb(c))
    plot.plotMap(data, lat=latC, lon=lonC, ax=ax, plotPoint=True,
                cbar=False, title='map of '+case)
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'case_{}_sample_map'.format(case))
    fig.savefig(saveFile)
