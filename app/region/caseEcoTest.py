import os
from hydroDL import pathSMAP
from hydroDL.utils.app import ecoReg_ind
from hydroDL.data import dbCsv
from hydroDL import master
from hydroDL.post import plot, stat
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgb

caseLstTup = ([
    '080305', '080300', '080305+090200', '080305+090300', '080305+090400',
    '080305+100200', '080305+060200'
], [
    '090303', '090300', '090303+090401', '090303+090402', '090303+100204',
    '090303+060200'
], [
    '090401', '090401+090402', '090401+090301', '090401+090203',
    '090401+100105', '090401+080305', '090401+100204'
])
saveFolder = os.path.join(pathSMAP['dirResult'], 'regionalization')

# test
tRange = [20160401, 20180401]
subsetPattern = 'ecoRegionL3_{}_v2f1'
rootDB = pathSMAP['DB_L3_NA']
dfC = dbCsv.DataframeCsv(rootDB=rootDB, subset='CONUSv2f1', tRange=tRange)
latC, lonC = dfC.getGeo()

for caseLst in caseLstTup:
    testName = subsetPattern.format(caseLst[0])
    errLst = list()
    for case in caseLst:
        outName = subsetPattern.format(case) + '_Forcing'
        out = os.path.join(pathSMAP['Out_L3_NA'], 'ecoRegionL3', outName)
        df, yp, yt = master.test(out, tRange=tRange, subset=testName)
        errLst.append(stat.statError(yp[:, :, 0], yt[:, :, 0]))

    # plot box
    keyLst = stat.keyLst
    dataBox = list()
    for key in keyLst:
        temp = list()
        for err in errLst:
            temp.append(err[key])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox,
                          keyLst,
                          caseLst,
                          title=caseLst[0],
                          figsize=(12, 4),
                          sharey=False)
    # plt.subplots_adjust(wspace=0, hspace=0)
    fig.show()
    saveFile = os.path.join(saveFolder, 'case_' + caseLst[0])
    fig.savefig(saveFile)

    # plot maps
    cLst = 'rbkgcmy'
    nCase = len(caseLst)
    fig, ax = plt.subplots(figsize=(8, 4))
    data = np.tile(np.array(to_rgb('lightgrey')), (latC.shape[0], 1))
    for k in range(1, nCase):
        case = caseLst[k]
        subsetLst = [subsetPattern.format(x) for x in case.split('+')]
        dfsub = dbCsv.DataframeCsv(rootDB=rootDB,
                                   subset=subsetLst,
                                   tRange=tRange)
        data[dfsub.indSub, :] = np.array(to_rgb(cLst[k]))
    data[df.indSub, :] = np.array(to_rgb('r'))
    plot.plotMap(data, lat=latC, lon=lonC, ax=ax, cbar=False)
    fig.show()
    saveFile = os.path.join(saveFolder, 'case_' + caseLst[0] + '_map')
    fig.savefig(saveFile)
